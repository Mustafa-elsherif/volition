"""Request and response models for the decoding API.

A trial is a set of neurons. Each neuron carries its z-scored firing rate in the
pre-movement window and its Beryl brain-region acronym. The client sends the same
normalized rates the model was trained on; the server does not re-normalize,
because per-neuron normalization needs the whole session and cannot be done from a
single trial.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Neuron(BaseModel):
    rate: float = Field(..., description="Z-scored firing rate in the pre-movement window.")
    region: str = Field(..., description="Beryl region acronym, for example VISp or CP.")


class DecodeRequest(BaseModel):
    neurons: list[Neuron] = Field(..., min_length=1, max_length=512)


class NeuronContribution(BaseModel):
    region: str
    rate: float
    attention: float = Field(..., description="Share of pooling attention, 0 to 1.")


class RegionContribution(BaseModel):
    region: str
    attention: float
    n_neurons: int


class DecodeResponse(BaseModel):
    probability: float = Field(..., description="Probability of choice = +1 (right).")
    choice: str = Field(..., description="Decoded choice label.")
    confidence: float = Field(..., description="Distance of the probability from 0.5, scaled to 0-1.")
    n_neurons: int
    top_neurons: list[NeuronContribution]
    region_contributions: list[RegionContribution]


class ChatTurn(BaseModel):
    role: str
    content: str


class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


class AssistantResponse(BaseModel):
    answer: str
    followups: list[str]
    grounded: bool
    source: str


class ModelInfo(BaseModel):
    name: str
    version: str
    task: str
    n_regions: int
    max_neurons: int
    metrics: dict
    known_regions: list[str]
