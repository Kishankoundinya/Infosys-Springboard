from fastapi import APIRouter
from backend.app.models.schemas import SettingsResponse, SettingsUpdateRequest
from backend.app.utils.config import settings

router = APIRouter(prefix="/api/settings", tags=["Settings"])

@router.get("", response_model=SettingsResponse)
def get_current_settings():
    return SettingsResponse(
        app_name=settings.APP_NAME,
        llm_model=settings.LLM_MODEL,
        device=settings.DEVICE,
        temperature=settings.TEMPERATURE,
        max_new_tokens=settings.MAX_NEW_TOKENS,
        embedding_model=settings.EMBEDDING_MODEL,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        top_k=settings.TOP_K,
        max_retrieval_steps=settings.MAX_RETRIEVAL_STEPS,
        max_file_size_mb=settings.MAX_FILE_SIZE_MB,
    )

@router.post("", response_model=SettingsResponse)
def update_settings(update: SettingsUpdateRequest):
    if update.temperature is not None:
        settings.TEMPERATURE = update.temperature
    if update.max_new_tokens is not None:
        settings.MAX_NEW_TOKENS = update.max_new_tokens
    if update.chunk_size is not None:
        settings.CHUNK_SIZE = update.chunk_size
    if update.chunk_overlap is not None:
        settings.CHUNK_OVERLAP = update.chunk_overlap
    if update.top_k is not None:
        settings.TOP_K = update.top_k
    if update.max_retrieval_steps is not None:
        settings.MAX_RETRIEVAL_STEPS = update.max_retrieval_steps

    return get_current_settings()
