# ADR 001: Use PostgreSQL with pgvector for Semantic Search

*   **Status:** Accepted
*   **Date:** 2023-10-27

## Context

The new hire onboarding platform requires a search functionality that goes beyond simple keyword matching. New hires need to find relevant information (e.g., documentation, FAQs, policies) based on the semantic meaning or intent of their queries, not just the exact words used. For example, a query for "how to set up my computer" should return results for "local development environment configuration."

This requires implementing a vector-based semantic search system. The core architectural decision is whether to:
1.  Integrate vector search capabilities into our existing primary database (PostgreSQL).
2.  Introduce a separate, specialized vector database (e.g., Weaviate, Milvus, ChromaDB) alongside our primary database.

The decision must consider implementation complexity, operational overhead, performance, data consistency, and the specific needs of our application, which include filtering search results by metadata such as department, role, or content tags. Our anticipated dataset size is in the thousands to low hundreds of thousands of documents, not billions.

## Decision

We will use our existing PostgreSQL database, extended with the `pgvector` extension, to store, index, and query vector embeddings for semantic search.

This approach involves adding a `vector` type column to our content tables (e.g., `articles`) and using `pgvector`'s functions and index types (e.g., HNSW, IVFFlat) to perform Approximate Nearest Neighbor (ANN) searches.

### Rationale

This decision was made over choosing a specialized vector database due to its operational simplicity, data integrity, and powerful querying capabilities, which are an excellent fit for our specific use case.

1.  **Unified Data Store & Simplified Architecture:** All data—the content text, its associated metadata, and the vector embedding—resides in a single database. This eliminates the significant architectural complexity of managing, securing, and operating two separate data systems.

2.  **Reduced Operational Overhead:** Our team is already proficient in managing PostgreSQL. By leveraging our existing infrastructure and expertise, we avoid the learning curve and operational burden of deploying, monitoring, and backing up a new, distinct database technology.

3.  **Powerful Hybrid Search:** This is the most compelling technical advantage for our use case. `pgvector` allows us to combine traditional, exact-match SQL `WHERE` clauses with ANN vector search in a single atomic query. This enables highly efficient pre-filtering. For example, we can execute a query like: "Find articles semantically similar to 'VPN setup', but only for the 'Engineering' department and created in the last 6 months." This is far more efficient than the multi-step process required by a separate vector database (fetch IDs from vector DB, then filter those IDs in the primary DB).

4.  **Guaranteed Data Consistency:** Using a single transactional database provides strong ACID guarantees. When a document is created, updated, or deleted, the changes to the content and its vector embedding can be performed in a single, atomic transaction. This completely eliminates the data synchronization challenges and potential for stale search results inherent in a two-database architecture.

5.  **Cost-Effectiveness:** We leverage our existing PostgreSQL instance, avoiding the direct cost of a managed specialized vector database service or the indirect infrastructure and operational costs of self-hosting one. While we may need to scale up our PostgreSQL instance with more RAM to accommodate the vector index, this is still more cost-effective than managing a second system.

A specialized vector database was rejected because its primary advantages—extreme performance at massive scale (billions of vectors) and native horizontal scalability—are unnecessary for our application's data volume. These benefits do not outweigh the significant disadvantages of increased architectural complexity, data synchronization challenges, and higher operational overhead.

## Consequences

**Positive:**
*   **Simplified Architecture:** The system remains a simple, two-tier application with a single relational database, making it easier to understand, develop, and maintain.
*   **Lower Operational Burden:** The team continues to manage a single, familiar database system, reducing costs for infrastructure, monitoring, and training.
*   **Strong Data Integrity:** The risk of data inconsistency between document metadata and vector embeddings is eliminated due to ACID transactions.
*   **Enhanced Feature Development:** The ability to perform efficient, powerful hybrid searches will allow us to build a superior, context-aware search experience for new hires with less engineering effort.
*   **Faster Time to Market:** Implementation is straightforward, involving a simple database extension and schema modification rather than integrating a new distributed system.

**Negative:**
*   **Potential for Resource Contention:** Computationally intensive vector searches will run on the same database instance as standard application queries (e.g., user logins, content updates). A high volume of search traffic could potentially impact the performance of other database operations. This will require careful monitoring and may necessitate scaling up the database instance.
*   **Scaling Limitations:** While PostgreSQL scales vertically very well, horizontal scaling is more complex than for a purpose-built distributed vector database. If our data volume were to unexpectedly grow into the hundreds of millions or billions of vectors, this decision would need to be re-evaluated.
*   **Dependency on Extension Features:** We are limited to the indexing algorithms and features provided by the `pgvector` extension, which may not be as advanced or tunable as those found in cutting-edge specialized databases.