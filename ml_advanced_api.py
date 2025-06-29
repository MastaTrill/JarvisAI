"""
HuggingFace Transformers and AutoML integration scaffold for Jarvis AI
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

# HuggingFace Transformers
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    pipeline = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

# Optuna for AutoML
try:
    import optuna
except ImportError:
    optuna = None

router = APIRouter()

class HFTextRequest(BaseModel):
    text: str
    model: Optional[str] = "distilbert-base-uncased-finetuned-sst-2-english"

@router.post("/ml/hf-text-classify")
def hf_text_classify(req: HFTextRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Transformers not installed")
    clf = pipeline("sentiment-analysis", model=req.model)
    result = clf(req.text)
    return {"result": result}


# AutoML endpoint: simple Optuna + LightGBM regression example
from typing import List
import numpy as np
import pandas as pd
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

class AutoMLTrainRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    n_trials: int = 10

@router.post("/ml/automl-train")
def automl_train(req: AutoMLTrainRequest):
    if optuna is None or lgb is None:
        raise HTTPException(status_code=503, detail="Optuna/LightGBM not installed")
    X = np.array(req.X)
    y = np.array(req.y)
    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'num_leaves': trial.suggest_int('num_leaves', 8, 128),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        }
        dtrain = lgb.Dataset(X, y)
        cv = lgb.cv(param, dtrain, nfold=3, stratified=False, seed=42, verbose_eval=False)
        return min(cv['rmse-mean'])
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=req.n_trials)
    return {"best_params": study.best_params, "best_value": study.best_value}
