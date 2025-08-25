from pydantic import BaseModel
from typing import List, Dict, Optional

class ProductRequirementsDocument(BaseModel):
    status: str
    author: str
    version: str
    last_updated: str
    executive_summary: str
    vision: str
    problem_statement: str
    user_personas: List[str]
    scenarios: Dict[str, str]
    goals: List[str]
    success_metrics: List[Dict[str, str]]
    functional_requirements: List[str]
    user_stories: List[Dict[str, List[Dict[str, List[str]]]]]
    non_functional_requirements: List[str]
    release_plan: List[Dict[str, str]]
    out_of_scope: List[str]
    future_considerations: List[str]
    appendix: Optional[List[str]]
    open_questions: Optional[List[str]]
    dependencies: Optional[List[str]]