from dataclasses import dataclass
from datetime import time
from typing import Optional


@dataclass
class Metadata:
    title: str
    page: int


@dataclass
class Document:
    page_content: str
    title: str
    metadata: dict
    embedding: Optional[list[float]] = None


@dataclass
class QueryResult:
    query: str
    contexts: list[str]
    prompt: str
    response: str
    evaluations: Optional[dict] = None


@dataclass
class ExperimentResults:
    results: list[QueryResult]
    model: str
    parameters: dict[str, any]
    timestamp_end: Optional[time]
    evaluations: Optional[dict] = None
