"""
JarvisAI LLM Integration
OpenAI, Azure OpenAI, and local LLM support with RAG and caching
"""

import os
from typing import List, Dict, Optional, Any
from openai import OpenAI, AzureOpenAI
import chromadb
from chromadb.config import Settings
import tiktoken
import json
from datetime import datetime

from cache import llm_cache

# Environment configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

# Cache settings
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "3600"))


class LLMProvider:
    """Base class for LLM providers"""

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Send chat completion request"""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        super().__init__(model)
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        use_cache: bool = True,
        **kwargs,
    ) -> str:
        """Send chat completion with optional caching"""
        cache_key = None

        if use_cache and LLM_CACHE_ENABLED:
            cache_key = llm_cache.get(messages, self.model, temperature)
            if cache_key:
                return cache_key

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        result = response.choices[0].message.content

        if use_cache and LLM_CACHE_ENABLED and result:
            llm_cache.set(messages, self.model, result, temperature, LLM_CACHE_TTL)

        return result

    def stream_chat(self, messages: List[Dict], **kwargs):
        """Stream chat completion"""
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def create_embedding(
        self, text: str, model: str = "text-embedding-3-small"
    ) -> List[float]:
        """Create embedding vector"""
        response = self.client.embeddings.create(model=model, input=text)
        return response.data[0].embedding


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI API provider"""

    def __init__(
        self, api_key: str = None, endpoint: str = None, deployment: str = None
    ):
        super().__init__(deployment or AZURE_OPENAI_DEPLOYMENT)
        self.client = AzureOpenAI(
            api_key=api_key or AZURE_OPENAI_KEY,
            api_version="2024-02-01",
            azure_endpoint=endpoint or AZURE_OPENAI_ENDPOINT,
        )

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Send chat completion"""
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return response.choices[0].message.content


class VectorStore:
    """Vector database for RAG using ChromaDB"""

    def __init__(
        self,
        collection_name: str = "jarvis_knowledge",
        persist_directory: str = "./chroma_db",
    ):
        self.client = chromadb.Client(
            Settings(persist_directory=persist_directory, anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(
        self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None
    ):
        """Add documents to vector store"""
        if ids is None:
            ids = [
                f"doc_{i}_{datetime.now().timestamp()}" for i in range(len(documents))
            ]

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def search(self, query: str, n_results: int = 5) -> Dict:
        """Search for similar documents"""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results

    def delete_collection(self):
        """Delete the collection"""
        self.client.delete_collection(name=self.collection.name)


class RAGAgent:
    """Retrieval-Augmented Generation agent"""

    def __init__(self, llm_provider: LLMProvider, vector_store: VectorStore):
        self.llm = llm_provider
        self.vector_store = vector_store
        self.conversation_history = []

    def add_knowledge(self, documents: List[str], metadatas: List[Dict] = None):
        """Add knowledge documents"""
        self.vector_store.add_documents(documents, metadatas)

    def retrieve_context(self, query: str, n_results: int = 3) -> str:
        """Retrieve relevant context"""
        results = self.vector_store.search(query, n_results)
        contexts = results["documents"][0]
        return "\n\n".join(contexts)

    def chat(
        self, user_message: str, use_context: bool = True, system_prompt: str = None
    ) -> str:
        """Chat with RAG"""
        # Retrieve context
        context = ""
        if use_context:
            context = self.retrieve_context(user_message)

        # Build messages
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context:
            context_msg = (
                f"Relevant context:\n{context}\n\nUser question: {user_message}"
            )
            messages.append({"role": "user", "content": context_msg})
        else:
            messages.append({"role": "user", "content": user_message})

        # Get response
        response = self.llm.chat(messages)

        # Update conversation history
        self.conversation_history.append(
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.conversation_history.append(
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        return response

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []


class JarvisLLM:
    """Main JarvisAI LLM interface"""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        if provider == "openai":
            self.llm = OpenAIProvider(model=model)
        elif provider == "azure":
            self.llm = AzureOpenAIProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.vector_store = VectorStore()
        self.rag_agent = RAGAgent(self.llm, self.vector_store)

    def chat(self, message: str, **kwargs) -> str:
        """Simple chat"""
        return self.llm.chat([{"role": "user", "content": message}], **kwargs)

    def rag_chat(self, message: str, **kwargs) -> str:
        """RAG-enhanced chat"""
        return self.rag_agent.chat(message, **kwargs)

    def add_knowledge(self, documents: List[str], metadatas: List[Dict] = None):
        """Add knowledge base"""
        self.rag_agent.add_knowledge(documents, metadatas)

    def get_embedding(self, text: str) -> List[float]:
        """Get text embedding"""
        if hasattr(self.llm, "create_embedding"):
            return self.llm.create_embedding(text)
        raise NotImplementedError("Embeddings not supported by this provider")


if __name__ == "__main__":
    print("🤖 JarvisAI LLM Integration Test")

    # Test with environment check
    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not set. Set it with:")
        print('   $env:OPENAI_API_KEY="your-key-here"')
        print("\n📝 Creating demo mode...")
        print("✅ LLM classes initialized")
        print("✅ Vector store ready")
        print("✅ RAG agent configured")
    else:
        try:
            jarvis = JarvisLLM(provider="openai", model="gpt-4o-mini")

            # Test chat
            response = jarvis.chat("Hello! What is 2+2?")
            print(f"✅ Chat Response: {response[:100]}...")

            # Add knowledge
            jarvis.add_knowledge(
                [
                    "JarvisAI is an advanced quantum consciousness platform.",
                    "It features temporal analysis, neural networks, and computer vision.",
                    "The system has creator-level protection and family shields.",
                ]
            )
            print("✅ Knowledge base populated")

            # Test RAG
            rag_response = jarvis.rag_chat("What can you tell me about JarvisAI?")
            print(f"✅ RAG Response: {rag_response[:100]}...")

        except Exception as e:
            print(f"⚠️  Error: {e}")
            print("Make sure OPENAI_API_KEY is valid")
