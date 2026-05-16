from pydantic import BaseModel
from typing import List, Dict, Optional


class PredictRequest(BaseModel):
    model_name: str
    data: List[List[float]]


__all__ = ["PredictRequest"]
