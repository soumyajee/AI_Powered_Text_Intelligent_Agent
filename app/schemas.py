from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    sentiment: str
    keywords: list[str]

class SummaryResponse(BaseModel):
    summary: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
