from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ObservationCreate(BaseModel):
    ville: str = Field(..., min_length=1, max_length=100)
    marche: str = Field(..., min_length=1, max_length=100)
    produit: str = Field(..., min_length=1, max_length=100)
    prix_unitaire: float = Field(..., gt=0)
    devise: str = Field(default="XAF", max_length=10)
    unite_mesure: str = Field(..., min_length=1, max_length=50)
    remarque: Optional[str] = None


class ObservationResponse(BaseModel):
    id: int
    timestamp: datetime
    ville: str
    marche: str
    produit: str
    prix_unitaire: float
    devise: str
    unite_mesure: str
    remarque: Optional[str]

    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    features: list[float]


class RegressionRequest(BaseModel):
    target: str
    predictors: list[str]


class ClusteringRequest(BaseModel):
    n_clusters: int = Field(default=3, ge=2, le=10)
    features: Optional[list[str]] = None


class KNNRequest(BaseModel):
    target: str
    k: int = Field(default=5, ge=1, le=20)
