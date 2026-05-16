# GitHub AI Agent

An AI-powered research agent that explores GitHub repositories to answer technical questions. Built with Django, Django REST Framework, and OpenAI's function calling API.

## Architecture

The agent uses **tool calling** to autonomously explore a repository's codebase:

1. **Receives a question** about a GitHub repo via REST API
2. **Clones/updates** the repo locally
3. **Checks prior findings** from previous sessions (avoids redundant work)
4. **Iteratively calls tools** (list files, read files, search code, get summaries) — the LLM decides what to explore
5. **Saves findings** to the database as it discovers important information
6. **Produces a final answer** grounded in the actual source code

### Agent Tools

| Tool | Purpose |
|------|---------|
| `list_files(path)` | List directory contents (capped at 200 entries) |
| `read_file(path, start_line, end_line)` | Read file contents (max 500 lines per call) |
| `search_code(query)` | Grep-style search across all files (top 20 matches) |
| `get_file_summary(path)` | File metadata; Python files include classes/functions/imports |
| `save_finding(file_path, note, finding_type)` | Persist a discovery for current and future sessions |
| `get_previous_findings(repo_url)` | Retrieve findings from prior sessions |
| `list_past_sessions(repo_url)` | List previous Q&A sessions |

### Safety Controls

- **Max 15 tool calls** per session (prevents infinite loops)
- **100k token budget** (aborts if exceeded)
- **Context window management** (trims old messages, keeps last 30)
- **File size caps** (skips files > 500KB, truncates output)

## Database Schema

```
Repository ──< ResearchSession ──< ToolCall
                                ──< Finding
```

- **Repository**: GitHub repos that have been researched
- **ResearchSession**: Each question asked, with answer, token usage, and timing
- **ToolCall**: Every tool invocation logged with input/output
- **Finding**: Semantic discoveries that persist across sessions

## Setup

### Prerequisites

- Python 3.12+
- An OpenAI API key

### Installation

You can install the project dependencies using either `pip` or `uv` (a much faster Python package installer).

#### Option A: Using pip

```bash
# Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Alternatively, to install as an editable package: pip install -e .
```

#### Option B: Using uv (Recommended for speed)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
# Alternatively, to install as an editable package: uv pip install -e .
```

#### Common Setup Steps

After installing dependencies using either method above, proceed with the following setup:

```bash
# Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# Run migrations
python manage.py migrate

# Seed sample data (no API key needed)
python manage.py seed_sample_data

# Start the server
python manage.py runserver
```

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `POST` | `/api/research/sessions/` | Start a new research session |
| `GET` | `/api/research/sessions/list/` | List past sessions (optional `?repo_url=` filter) |
| `GET` | `/api/research/sessions/<session_id>/` | Get session detail with tool calls & findings |
| `GET` | `/api/repositories/` | List all researched repositories |
| `GET` | `/api/repositories/<id>/` | Get repository detail |

### Swagger UI

Visit `http://localhost:8000/api/docs/` for interactive API documentation.

### Example: Start a Research Session

```bash
curl -X POST http://localhost:8000/api/research/sessions/ \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/pallets/flask",
    "question": "How does Flask handle routing?"
  }'
```

## Running Tests

```bash
python manage.py test app -v2
```

## Generating Sample Data

```bash
# Mock data (no API key needed)
python manage.py seed_sample_data

# Live agent run (requires OPENAI_API_KEY)
python manage.py seed_sample_data --live \
  --repo-url https://github.com/pallets/flask \
  --question "How does Flask handle routing?"
```

## Design Decisions

### Why tool calling over RAG?
The original codebase used a linear RAG pipeline (embed entire repo → vector search → answer). The new design gives the LLM autonomy to decide what to explore, which is more cost-efficient (no upfront embedding cost) and produces better answers (targeted file reading vs. similarity-guessed chunks).

### Why separate ToolCall and Finding tables?
ToolCalls are mechanical logs (every invocation). Findings are semantic discoveries the agent explicitly saves. They have different query patterns — you might want "all findings about auth in repo X" without wading through hundreds of `list_files` calls.

### Why denormalized counters on ResearchSession?
`total_tool_calls`, `total_findings`, and `total_tokens` on the session enable quick dashboard queries without JOINing to child tables.

### Context management
- Files > 500 lines: agent reads in ranges via `start_line`/`end_line`
- Directories > 200 entries: truncated with a note
- Messages > 30: old messages trimmed to stay within context window
- Token budget: hard cap at 100k tokens per session

## OpenAI Setup

Create a local `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4.1
OPENAI_EXTRACTOR_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```
