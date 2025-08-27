import datetime
from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class GoalMetric(BaseModel):
    """Represents a single goal with its corresponding KPI and target."""
    goal: str = Field(..., description="The high-level objective, e.g., 'Improve New Hire Efficiency'.")
    kpi: str = Field(..., alias="Key Performance Indicator (KPI)", description="The metric used to measure progress towards the goal.")
    target: str = Field(..., description="The specific, measurable target for the KPI, e.g., 'Decrease by 20% in Q1'.")

    class Config:
        allow_population_by_field_name = True


class UserStory(BaseModel):
    """Represents a single user story with its acceptance criteria."""
    title: str = Field(..., description="The user story in the format: 'As a [persona], I want to [action], so that [benefit]'.")
    acceptance_criteria: List[str] = Field(..., description="A list of conditions that must be met for the story to be considered complete.")


class Epic(BaseModel):
    """Represents a collection of related user stories."""
    title: str = Field(..., description="The name of the epic, e.g., 'User Authentication'.")
    stories: List[UserStory] = Field(..., description="A list of user stories belonging to this epic.")


class Milestone(BaseModel):
    """Represents a single milestone in the release plan."""
    version: str = Field(..., description="The version name for the milestone, e.g., 'Version 1.0 (MVP)'.")
    target_date: str = Field(..., description="The target delivery date for this milestone.")
    description: str = Field(..., description="A summary of the features and capabilities included in this milestone.")


class ProductRequirementsDocument(BaseModel):
    """
    A Pydantic model representing the structure of a Product Requirements Document (PRD).
    """
    product_name: str = Field(..., description="The name of the product this document describes.")
    status: Literal["Draft", "In Review", "Approved"] = Field(..., description="The current status of the PRD.")
    author: str = Field(..., description="The name of the author or team responsible for the PRD.")
    version: str = Field(..., example="1.0", description="The document version.")
    last_updated: datetime.date = Field(..., description="The date the document was last updated.")

    executive_summary_and_vision: str = Field(..., description="A high-level overview of the product, its purpose, the problem it solves, and its ultimate vision.")

    # Section 2: The Problem
    problem_statement: str = Field(..., description="A clear and concise description of the primary problem.")
    user_personas_and_scenarios: List[str] = Field(..., description="A list of key user personas affected by the problem and their scenarios.")

    # Section 3: Goals & Success Metrics
    goals_and_success_metrics: List[GoalMetric] = Field(..., description="A list of specific, measurable outcomes and KPIs for success.")

    # Section 4: Functional Requirements & User Stories
    functional_requirements: List[Epic] = Field(..., description="A list of epics, each containing user stories that detail what the product must do.")

    # Section 5: Non-Functional Requirements (NFRs)
    non_functional_requirements: Dict[str, str] = Field(..., description="A dictionary of the system's quality attributes, such as Performance, Security, Accessibility, and Scalability.")

    # Section 6: Release Plan & Milestones
    release_plan_and_milestones: List[Milestone] = Field(..., description="A high-level timeline for delivery, broken into versioned milestones.")

    # Section 7: Out of Scope & Future Considerations
    out_of_scope_for_v1: List[str] = Field(..., description="A list of features or functionalities explicitly not being built in the initial version.")
    future_work: List[str] = Field(..., description="A list of potential features or integrations for future consideration.")

    # Section 8: Appendix & Open Questions
    appendix_and_open_questions: List[str] = Field(..., description="A list of dependencies, assumptions, and questions that need answers.")