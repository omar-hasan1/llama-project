from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.inference_service import get_inference

router = APIRouter()

class InferenceRequest(BaseModel):
    text: str

@router.post("/ask")
async def ask(request: InferenceRequest):
    try:
        response = get_inference(request.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
