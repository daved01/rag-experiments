from dataclasses import dataclass
from typing import Optional
from datetime import time


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


@dataclass
class ExperimentResults:
    results: list[QueryResult]
    model: str
    parameters: dict[str, any]
    timestamp_end: Optional[time]
