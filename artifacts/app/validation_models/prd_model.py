from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union
from datetime import date


class Persona(BaseModel):
    """Represents a user persona with their name and a typical scenario."""
    name: str = Field(..., description="The name of the user persona (e.g., The New Hire).")
    scenario: str = Field(..., description="A typical scenario or situation faced by this persona that highlights the problem.")


class GoalMetric(BaseModel):
    """Represents a product goal with its Key Performance Indicator (KPI) and target."""
    goal: str = Field(..., description="A specific, measurable goal for the product.")
    kpi: str = Field(..., description="The Key Performance Indicator used to measure the progress towards the goal.")
    target: str = Field(..., description="The measurable target value or outcome for the KPI.")


class UserStory(BaseModel):
    """Represents a functional requirement as a user story with acceptance criteria."""
    epic: Optional[str] = Field(None, description="The epic or larger theme this user story belongs to (e.g., User Authentication).")
    story: str = Field(..., description="The user story, typically in the format: 'As a [user], I want to [action], so that [benefit]'.")
    acceptance_criteria: List[str] = Field(..., description="A list of conditions that must be met for the user story to be considered complete.")


class ReleaseMilestone(BaseModel):
    """Represents a planned release version with its target date and core features."""
    version: str = Field(..., description="The version number or identifier of the release (e.g., 1.0, MVP).")
    target_date: Union[date, str] = Field(..., description="The target date for the release (can be a specific date or a descriptive string like 'Q1').")
    description: str = Field(..., description="A summary of the core features or scope included in this release.")


class ProductRequirementsDocument(BaseModel):
    """
    Pydantic model for validating the structure of a Product Requirements Document (PRD).
    This model represents the full PRD template, ensuring all key sections and their
    expected data types are present and correctly formatted.
    """

    # --- Header / Metadata ---
    product_name: str = Field(..., description="The name of the product this PRD describes, taken from the document title.")
    status: str = Field(..., description="The current status of the PRD (e.g., Draft, Approved, In Review).")
    author: str = Field(..., description="The name or team responsible for authoring the PRD.")
    version: str = Field(..., description="The version number of the PRD document itself.")
    last_updated: date = Field(..., description="The date when the PRD was last updated.")

    # --- 1. Executive Summary & Vision ---
    executive_summary_vision: str = Field(..., description="A high-level overview for stakeholders, describing the product's purpose, core problem solved, target users, and ultimate vision.")

    # --- 2. The Problem ---
    problem_statement: str = Field(..., description="A clear and concise description of the primary problem the product aims to solve.")
    user_personas_scenarios: List[Persona] = Field(..., description="A list of key user personas affected by the problem, each with a typical scenario highlighting the pain point.")

    # --- 3. Goals & Success Metrics ---
    goals_success_metrics: List[GoalMetric] = Field(..., description="A list of specific, measurable goals for the product, along with their Key Performance Indicators (KPIs) and target values.")

    # --- 4. Functional Requirements & User Stories ---
    functional_requirements_user_stories: List[UserStory] = Field(..., description="The core of the PRD, detailing what the product must do, broken down into actionable user stories with acceptance criteria.")

    # --- 5. Non-Functional Requirements (NFRs) ---
    non_functional_requirements: Dict[str, str] = Field(..., description="A dictionary where keys are categories of non-functional requirements (e.g., 'Performance', 'Security') and values are their detailed descriptions.")

    # --- 6. Release Plan & Milestones ---
    release_plan_milestones: List[ReleaseMilestone] = Field(..., description="A high-level timeline for product delivery, outlining different versions and their core features and target dates.")

    # --- 7. Out of Scope & Future Considerations ---
    out_of_scope_v1: List[str] = Field(..., description="A list of features or functionalities explicitly out of scope for the initial version (V1.0) to manage expectations and prevent scope creep.")
    future_work: List[str] = Field(..., description="A list of potential features, integrations, or work planned for future iterations beyond V1.0.")

    # --- 8. Appendix & Open Questions ---
    open_questions: List[str] = Field(..., description="A list of open questions, decisions, or clarifications that need to be addressed.")
    dependencies: List[str] = Field(..., description="A list of external dependencies, assumptions, or prerequisites for the project's success.")