# Design Decisions & Architecture (DECISIONS.md)

## Architecture Overview

The GitHub Research AI Agent is structured as a Django REST Framework (DRF) backend orchestrating a LangGraph-powered state machine. 

At a high level, the system is separated into three primary layers:
1. **API / Transport Layer (`views.py` & `serializers.py`)**: Responsible strictly for HTTP routing, request validation, and JSON serialization. Complex business logic (like URL extraction) was decoupled into dedicated utilities.
2. **Service Layer (`research_service.py` & `github_service.py`)**: Acts as the orchestrator. It manages the lifecycle of a research request—cloning or pulling the GitHub repository, initializing database records, and triggering the agent.
3. **Agent Layer (`repository_agent.py` & `tools.py`)**: Uses **LangGraph** to model the research process as a deterministic State Graph. The Language Model (LLM) iteratively calls tools (like searching code, reading files, and viewing directory structures) to build up a context window until it reaches a conclusive answer, which is parsed via a structured Pydantic schema (`FinalAnswer`).

## Database Schema Rationale

The database is modeled around four primary entities: `Repository`, `ResearchSession`, `ToolCall`, and `Finding`. 

**Rationale:**
- **`ResearchSession` as the root entity**: Every question asked about a repo creates a unique session. It tracks metadata like token usage, execution duration, and completion status. This denormalized tracking is critical for billing and performance monitoring.
- **`ToolCall` logging**: We explicitly log every tool invocation (input, output, and duration) tied to a session. This is invaluable for observability, allowing developers to replay or audit the exact "train of thought" the AI took to arrive at an answer.

**Trade-offs & Scaling Considerations:**
Currently, `ToolCall` outputs are saved directly as text fields. At scale, storing massive string blobs of file contents in a relational database like PostgreSQL will lead to serious database bloat and performance degradation. 
*At scale*, I would offload raw tool output logs to a cheaper, scalable object store (like AWS S3) or a NoSQL document database (like MongoDB/Elasticsearch), keeping only lightweight metadata and references in PostgreSQL.

## Key Design Decisions & Trade-offs

1. **Synchronous Execution vs. Background Workers**:
   - *Decision*: The API currently blocks synchronously while the agent runs, returning the final populated session in the HTTP response. 
   - *Trade-off*: While this simplifies the frontend experience (no polling required), it is a major anti-pattern for production web servers. Complex queries taking 1-2 minutes will result in `504 Gateway Timeouts` behind proxies like Cloudflare or Nginx.

2. **Native OS Tools over Python Iteration (`git grep`)**:
   - *Decision*: Replaced a recursive Python directory search (`Path.rglob`) with a subprocess call to native `git grep`.
   - *Trade-off*: We sacrifice strict cross-platform purity (requires `git` installed on the host OS), but the performance gain is orders of magnitude faster and automatically respects complex `.gitignore` rules without custom logic.

3. **Context Injection via `RunnableConfig`**:
   - *Decision*: Tools are implemented as standalone functions instead of closures. Repository paths and DB session IDs are injected at runtime via LangChain's `RunnableConfig`.
   - *Trade-off*: Requires slightly more complex graph invocation syntax, but it guarantees that the LangGraph state remains serializable, which is a hard prerequisite for persistent checkpointer storage and distributed worker setups.

## What I'd Do Differently With More Time

1. **Implement Celery + WebSockets/SSE**: I would re-architect the service layer to immediately return a `202 Accepted` with a task ID. A Celery worker would process the LangGraph execution in the background, streaming updates (e.g., "Searching codebase...", "Reading models.py...") to the frontend via Server-Sent Events (SSE).
2. **Vector/Semantic Search (RAG)**: `git grep` is fantastic for exact matches, but fails at semantic queries. I would integrate a lightweight local vector database (like ChromaDB) to embed the repository codebase upon cloning, allowing the agent to perform semantic semantic similarity searches alongside regex searches.
3. **Robust Token Management**: The current sliding-window truncation drops older messages when the context limit is reached. With more time, I would implement a summarization node that condenses older tool findings into a dense "scratchpad" rather than outright deleting them, preserving critical early context.

## AI Coding Tools Usage

At first, I carefully reviewed the requirements and created an implementation plan before starting development. I primarily used AI coding tools, especially Antigravity, to accelerate the initial project scaffolding and implementation process. My general workflow when using AI agents is to first establish a solid boilerplate and architecture foundation, then iteratively refine and validate the generated code manually.

I initially instructed the AI agent to generate the complete implementation based on the provided requirements. The generated version created a `StartResearchSessionView` API where `git_repo` and `question` were passed as separate input fields. After reviewing the implementation, I decided to improve the developer and user experience by redesigning the API to accept a single input string containing both the GitHub repository URL and the question.

I manually guided the AI agent to update the implementation so that:
* the system accepts a single input string,
* extracts the GitHub repository URL using regex,
* separates the repository URL from the question text,
* and falls back to using a low-cost OpenAI model if regex extraction fails.

I also added validation logic so that:
* if no valid GitHub repository URL is found, the API returns a proper error response,
* and if a repository is found, it is passed correctly into the research workflow.

After the core functionality was updated, I instructed the AI agent to refactor and clean the codebase to improve readability, structure, and maintainability. However, I manually reviewed all critical areas, especially:
* request validation,
* LangGraph orchestration flow,
* database interactions,
* and repository parsing logic.

During manual API testing, I noticed the response handling was implemented asynchronously. Since this was a time-bound assessment task and immediate response behavior was more practical for evaluation purposes, I manually modified the implementation to use synchronous response handling instead.

Finally, I manually tested the APIs end-to-end to verify:
* repository extraction,
* question parsing,
* research session creation,
* LangGraph execution flow,
* and final response generation.

Overall, AI coding tools significantly accelerated scaffolding, repetitive implementation tasks, and refactoring suggestions. However, architectural decisions, debugging, validation flow, API behavior adjustments, and final quality control were handled manually.
