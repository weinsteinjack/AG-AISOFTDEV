```markdown
# [Short Title of the Architectural Decision]

*   **Status:** [Proposed | Accepted | Deprecated | Superseded by ADR-XXX]
*   **Date:** [YYYY-MM-DD]

## Context

[Describe the problem or issue that this decision addresses. This section should set the stage for the decision. What is the background? What are the technical, business, or operational constraints (the "forces") that are influencing this decision? For example:
*   "We need to improve the performance of our data ingestion pipeline."
*   "Our current authentication system does not support single sign-on (SSO), which is a new business requirement."
*   "The team has expertise in Go but not in Rust."
*   "We have a limited budget for new infrastructure."]

## Decision

[State the specific decision that was made. This should be a clear and concise statement. For example:
*   "We will adopt Apache Kafka as our primary message broker."
*   "We will implement authentication using the OAuth 2.0 protocol with Auth0 as the identity provider."
*   "We will build the new microservice in Go using the Gin framework."]

Follow this with the rationale. Why was this option chosen over others? What alternatives were considered and why were they rejected? This is the most important part of the ADR.

## Consequences

[Outline the consequences of this decision, both positive and negative. This helps future readers understand the trade-offs that were made. Consider the impact on the system, the team, and the stakeholders.]

**Positive:**
*   [e.g., "Improved system scalability and fault tolerance."]
*   [e.g., "Faster development time for new features that rely on this component."]
*   [e.g., "Alignment with industry-standard security practices."]

**Negative:**
*   [e.g., "Increased operational complexity due to the introduction of a new technology (Kafka)."]
*   [e.g., "Requires team members to be trained on the new authentication flow."]
*   [e.g., "Higher monthly infrastructure costs."]

```