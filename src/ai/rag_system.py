"""
Document RAG (Retrieval-Augmented Generation) System for JarvisAI.

Supports:
- Upload documents (PDF, TXT, MD, HTML, CSV)
- Automatic text extraction and chunking
- TF-IDF based semantic search (no external embeddings needed)
- Query documents through chat interface
- Document management (list, delete, re-index)
"""

import os
import re
import math
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Dict, List, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Depends
from pydantic import BaseModel, Field

# Need Text and Column types
from sqlalchemy import Column, Integer, String, DateTime, Text

from src.infra.db_config import Base as ConfigBase
from src.infra.database import get_db
from src.infra.auth_helpers import get_current_user
from src.infra.models_user import User


# --- Database Models ---

class Document(ConfigBase):
    __tablename__ = "rag_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(20), nullable=False)
    file_size = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="processing")  # processing, ready, error
    error_message = Column(String(1000), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_by = Column(String(100))
    tags = Column(String(1000), default="")


class DocumentChunk(ConfigBase):
    __tablename__ = "rag_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    word_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# --- Pydantic Schemas ---

class DocumentInfo(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    chunk_count: int
    status: str
    created_at: str
    tags: str


class QueryRequest(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResult(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_chunks_searched: int


# --- Router ---

router = APIRouter(prefix="/rag", tags=["Document RAG"])


# --- Text Processing ---

CHUNK_SIZE = 512  # words per chunk
CHUNK_OVERLAP = 50  # words of overlap

SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm", ".csv", ".pdf", ".py", ".js", ".ts", ".json", ".xml", ".yaml", ".yml", ".log"}


def _extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from various file formats."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(file_bytes)
    elif ext in (".html", ".htm"):
        return _extract_html(file_bytes)
    elif ext == ".csv":
        return _extract_csv(file_bytes)
    else:
        # Plain text and code files
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1")


def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF using PyPDF2 if available, else fallback."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text.strip())
        return "\n\n".join(texts)
    except ImportError:
        # Fallback: try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                texts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        texts.append(text.strip())
                return "\n\n".join(texts)
        except ImportError:
            return "[PDF text extraction requires PyPDF2 or pdfplumber. Install with: pip install PyPDF2]"


def _extract_html(file_bytes: bytes) -> str:
    """Extract text from HTML."""
    try:
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text = []
                self.skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self.skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self.skip = False

            def handle_data(self, data):
                if not self.skip:
                    stripped = data.strip()
                    if stripped:
                        self.text.append(stripped)

        parser = TextExtractor()
        parser.feed(file_bytes.decode("utf-8", errors="ignore"))
        return "\n".join(parser.text)
    except Exception:
        return file_bytes.decode("utf-8", errors="ignore")


def _extract_csv(file_bytes: bytes) -> str:
    """Extract text from CSV."""
    import csv
    from io import StringIO

    text = file_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(StringIO(text))
    rows = []
    for row in reader:
        rows.append(" | ".join(row))
    return "\n".join(rows)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks by sentences."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_words + word_count > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap
            overlap_words = 0
            overlap_chunk = []
            for s in reversed(current_chunk):
                wc = len(s.split())
                if overlap_words + wc > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_words += wc
            current_chunk = overlap_chunk
            current_words = overlap_words

        current_chunk.append(sentence)
        current_words += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# --- TF-IDF Search ---

class TfidfIndex:
    """Simple in-memory TF-IDF index for document search."""

    def __init__(self):
        self.chunk_ids: List[int] = []
        self.chunk_texts: List[str] = []
        self.doc_ids: List[int] = []
        self.idf: Dict[str, float] = {}
        self.tf_idf_vectors: List[Dict[str, float]] = []
        self.vocabulary: set = set()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z][a-z0-9_]+\b', text)
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
            'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'just', 'because', 'but', 'and', 'or', 'if', 'while', 'about', 'up',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
            'their', 'what', 'which', 'who', 'whom',
        }
        return [t for t in tokens if t not in stop_words and len(t) > 2]

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build TF-IDF index from chunks."""
        self.chunk_ids = [c["id"] for c in chunks]
        self.chunk_texts = [c["content"] for c in chunks]
        self.doc_ids = [c["document_id"] for c in chunks]

        if not self.chunk_ids:
            return

        # Build document frequency
        df: Dict[str, int] = {}
        tokenized_docs: List[List[str]] = []

        for text in self.chunk_texts:
            tokens = self._tokenize(text)
            tokenized_docs.append(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1
            self.vocabulary.update(unique_tokens)

        n = len(self.chunk_ids)
        # Compute IDF
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log((n + 1) / (freq + 1)) + 1

        # Compute TF-IDF vectors
        self.tf_idf_vectors = []
        for tokens in tokenized_docs:
            tf: Dict[str, float] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            # Normalize TF
            max_tf = max(tf.values()) if tf else 1
            tf_idf: Dict[str, float] = {}
            for token, count in tf.items():
                normalized_tf = 0.5 + 0.5 * (count / max_tf)
                idf_val = self.idf.get(token, 0)
                tf_idf[token] = normalized_tf * idf_val
            self.tf_idf_vectors.append(tf_idf)

    def search(self, query: str, top_k: int = 5, document_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Search for most relevant chunks using cosine similarity."""
        if not self.chunk_ids:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Build query TF-IDF vector
        query_tf: Dict[str, float] = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
        max_tf = max(query_tf.values()) if query_tf else 1
        query_vec: Dict[str, float] = {}
        for token, count in query_tf.items():
            normalized_tf = 0.5 + 0.5 * (count / max_tf)
            idf_val = self.idf.get(token, 0)
            query_vec[token] = normalized_tf * idf_val

        # Compute cosine similarity
        query_magnitude = math.sqrt(sum(v ** 2 for v in query_vec.values()))
        if query_magnitude == 0:
            return []

        scores: List[tuple] = []
        for i, vec in enumerate(self.tf_idf_vectors):
            # Filter by document_ids if specified
            if document_ids and self.doc_ids[i] not in document_ids:
                continue

            dot_product = sum(query_vec.get(token, 0) * vec.get(token, 0) for token in query_tokens)
            vec_magnitude = math.sqrt(sum(v ** 2 for v in vec.values()))
            if vec_magnitude == 0:
                continue
            similarity = dot_product / (query_magnitude * vec_magnitude)
            if similarity > 0:
                scores.append((i, similarity))

        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = scores[:top_k]

        results = []
        for idx, score in top_results:
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "document_id": self.doc_ids[idx],
                "content": self.chunk_texts[idx][:500],
                "score": round(score, 4),
            })

        return results


# Global index instance
_tfidf_index = TfidfIndex()
_index_built = False


def _rebuild_index(db):
    """Rebuild the TF-IDF index from all chunks in the database."""
    global _index_built
    chunks = db.query(DocumentChunk).all()
    chunk_data = [{"id": c.id, "document_id": c.document_id, "content": c.content} for c in chunks]
    _tfidf_index.build_index(chunk_data)
    _index_built = True


# --- Endpoints ---

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    tags: str = Form(""),
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload and index a document for RAG."""
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(400, "File too large. Maximum size is 50MB.")

    # Create document record
    doc = Document(
        filename=filename,
        file_type=ext.lstrip("."),
        file_size=file_size,
        status="processing",
        created_by=current_user.username,
        tags=tags,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        # Extract text
        text = _extract_text(file_bytes, filename)

        if not text.strip():
            doc.status = "error"
            doc.error_message = "No text could be extracted from the file."
            db.commit()
            return {"message": "Uploaded but no text extracted", "document_id": doc.id, "status": "error"}

        # Chunk text
        chunks = _chunk_text(text)

        # Store chunks
        for i, chunk_content in enumerate(chunks):
            chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=i,
                content=chunk_content,
                word_count=len(chunk_content.split()),
            )
            db.add(chunk)

        doc.chunk_count = len(chunks)
        doc.status = "ready"
        db.commit()

        # Rebuild index
        _rebuild_index(db)

        return {
            "message": f"Document '{filename}' uploaded and indexed successfully",
            "document_id": doc.id,
            "chunks": len(chunks),
            "status": "ready",
        }
    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)[:1000]
        db.commit()
        raise HTTPException(500, f"Error processing document: {e}")


@router.post("/query")
def query_documents(
    body: QueryRequest,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Query indexed documents using semantic search."""
    global _index_built

    if not _index_built:
        _rebuild_index(db)

    results = _tfidf_index.search(body.query, top_k=body.top_k, document_ids=body.document_ids)

    # Enrich with document info
    enriched = []
    for r in results:
        doc = db.query(Document).filter_by(id=r["document_id"]).first()
        enriched.append({
            "chunk_id": r["chunk_id"],
            "document_id": r["document_id"],
            "filename": doc.filename if doc else "Unknown",
            "content": r["content"],
            "score": r["score"],
        })

    return {
        "query": body.query,
        "results": enriched,
        "total_chunks_searched": len(_tfidf_index.chunk_ids),
    }


@router.get("/documents")
def list_documents(
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all indexed documents."""
    query = db.query(Document)
    if status:
        query = query.filter_by(status=status)
    docs = query.order_by(Document.created_at.desc()).limit(limit).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_type": d.file_type,
            "file_size": d.file_size,
            "chunk_count": d.chunk_count,
            "status": d.status,
            "tags": d.tags,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a document and its chunks."""
    doc = db.query(Document).filter_by(id=document_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    # Delete chunks
    db.query(DocumentChunk).filter_by(document_id=document_id).delete()
    db.delete(doc)
    db.commit()

    # Rebuild index
    _rebuild_index(db)

    return {"message": f"Document '{doc.filename}' deleted"}


@router.post("/reindex")
def reindex_all(
    db=Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Force rebuild the entire search index."""
    _rebuild_index(db)
    return {"message": "Index rebuilt", "total_chunks": len(_tfidf_index.chunk_ids)}
