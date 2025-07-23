from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import date

class ProductRequirementsDocument(BaseModel):
    product_name: str
    status: str
    author: str
    version: str
    last_updated: date
    
    executive_summary_vision: str
    
    problem_statement: str
    user_personas_scenarios: List[str]
    
    goals_success_metrics: List[Dict[str, str]]
    
    functional_requirements_user_stories: str
    
    non_functional_requirements: List[str]
    
    release_plan_milestones: List[Dict[str, str]]
    
    out_of_scope_v1: List[str]
    future_work: List[str]
    
    open_questions: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None