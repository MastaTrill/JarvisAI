from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from src.ai.multimodal_ai import MultimodalAI

router = APIRouter()
mm_ai = MultimodalAI()

@router.post("/advanced/multimodal/infer", tags=["Advanced"], summary="Multimodal AI inference")
async def multimodal_infer(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
):
    audio_data = None
    image_data = None
    video_data = None
    if audio:
        audio_data = await audio.read()
    if image:
        image_data = await image.read()
    if video:
        video_data = await video.read()
    result = mm_ai.process(
        text=text,
        audio=audio_data,
        image=image_data,
        video=video_data,
    )
    return result