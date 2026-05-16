"""
Agent tools for repository exploration and database interaction.

These are standard @tool functions that accept a LangChain RunnableConfig
to receive the context (repo_path, session_id) injected by the graph.
"""
from __future__ import annotations

import ast
import logging
import re
import subprocess
import time
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.models import Finding, ResearchSession, ToolCall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_FILE_LINES = 500          # Max lines returned by read_file per call
MAX_DIR_ENTRIES = 200         # Max entries returned by list_files
MAX_SEARCH_RESULTS = 20       # Max matches from search_code
SEARCH_CONTEXT_LINES = 3      # Lines of context around each match
TOOL_OUTPUT_TRUNCATE = 10_000  # Truncate tool output stored in DB

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox",
    ".eggs", "*.egg-info",
}

def _is_skippable(path: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith(".") for part in path.parts)

# ---------------------------------------------------------------------------
# Context and Logging Helpers
# ---------------------------------------------------------------------------

def _get_context(config: RunnableConfig) -> tuple[Path, ResearchSession]:
    """Extract repository root and session from RunnableConfig."""
    configurable = config.get("configurable", {})
    repo_path = configurable.get("repo_local_path")
    session_id = configurable.get("session_id")
    
    if not repo_path or not session_id:
        raise ValueError("Missing 'repo_local_path' or 'session_id' in RunnableConfig")
        
    return Path(repo_path), ResearchSession.objects.get(pk=session_id)

def _log_tool_call(
    session: ResearchSession,
    tool_name: str,
    tool_input: dict,
    tool_output: str,
    duration_ms: float,
) -> None:
    """Persist a ToolCall record to the database."""
    # To maintain a step counter, we can just use the current count
    step_number = ToolCall.objects.filter(session=session).count() + 1
    
    ToolCall.objects.create(
        session=session,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output[:TOOL_OUTPUT_TRUNCATE] if tool_output else "",
        step_number=step_number,
        duration_ms=round(duration_ms, 2),
    )

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def list_files(path: str, config: RunnableConfig) -> str:
    """List files and directories at the given path relative to the repository root. Returns up to 200 entries."""
    start = time.perf_counter()
    root, session = _get_context(config)
    
    # Defaults to "." if empty
    path = path or "."
    target = root / path
    
    if not target.exists():
        result = f"Error: Path '{path}' does not exist."
    elif not target.is_dir():
        result = f"Error: '{path}' is a file, not a directory. Use read_file instead."
    else:
        entries = []
        try:
            for entry in sorted(target.iterdir()):
                rel = entry.relative_to(root).as_posix()
                if _is_skippable(entry.relative_to(root)):
                    continue
                kind = "dir" if entry.is_dir() else "file"
                size = ""
                if entry.is_file():
                    try:
                        size = f" ({entry.stat().st_size} bytes)"
                    except OSError:
                        pass
                entries.append(f"[{kind}] {rel}{size}")
                if len(entries) >= MAX_DIR_ENTRIES:
                    entries.append(f"... truncated at {MAX_DIR_ENTRIES} entries")
                    break
        except PermissionError:
            result = f"Error: Permission denied reading '{path}'."
        else:
            result = "\n".join(entries) if entries else "(empty directory)"
            
    duration = (time.perf_counter() - start) * 1000
    _log_tool_call(session, "list_files", {"path": path}, result, duration)
    return result

@tool
def read_file(path: str, start_line: int, end_line: int, config: RunnableConfig) -> str:
    """Read a file's contents from start_line to end_line (1-indexed, inclusive). Max 500 lines per call."""
    start = time.perf_counter()
    root, session = _get_context(config)
    target = root / path
    
    if not target.exists():
        result = f"Error: File '{path}' does not exist."
    elif target.is_dir():
        result = f"Error: '{path}' is a directory. Use list_files instead."
    else:
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total = len(lines)
            # Clamp range
            s = max(1, start_line) - 1
            e = min(s + MAX_FILE_LINES, end_line, total)
            selected = lines[s:e]
            header = f"File: {path} (lines {s+1}-{e} of {total})\n"
            if e < total:
                header += f"(showing {e - s} of {total} lines — request a different range to see more)\n"
            numbered = [f"{i+s+1:>4}: {line}" for i, line in enumerate(selected)]
            result = header + "\n".join(numbered)
        except Exception as exc:
            result = f"Error reading '{path}': {exc}"
            
    duration = (time.perf_counter() - start) * 1000
    _log_tool_call(
        session, "read_file",
        {"path": path, "start_line": start_line, "end_line": end_line},
        result, duration,
    )
    return result

@tool
def search_code(query: str, config: RunnableConfig) -> str:
    """Search for a text pattern across all files in the repository using git grep. Fast and accurate."""
    start = time.perf_counter()
    root, session = _get_context(config)
    
    try:
        # We use git grep as it's typically a git repository
        # -I (ignore binary), -n (line number), -C 2 (context)
        # Using subprocess for speed rather than python loops
        proc = subprocess.run(
            ["git", "grep", "-I", "-i", "-n", "-C", str(SEARCH_CONTEXT_LINES), query],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if proc.returncode == 0 and proc.stdout:
            # We limit the output to prevent massive context explosion
            output_lines = proc.stdout.splitlines()
            # If there are way too many lines, truncate
            MAX_OUTPUT_LINES = MAX_SEARCH_RESULTS * (SEARCH_CONTEXT_LINES * 2 + 2)
            if len(output_lines) > MAX_OUTPUT_LINES:
                output_lines = output_lines[:MAX_OUTPUT_LINES]
                output_lines.append(f"\n... (truncated to first {MAX_OUTPUT_LINES} lines of matches) ...")
            
            result = f"Matches for '{query}':\n\n" + "\n".join(output_lines)
        elif proc.returncode == 1:
            result = f"No matches found for '{query}'."
        else:
            result = f"Search error (code {proc.returncode}): {proc.stderr}"
            
    except Exception as e:
        result = f"Error executing search: {e}"

    duration = (time.perf_counter() - start) * 1000
    _log_tool_call(session, "search_code", {"query": query}, result, duration)
    return result

@tool
def get_file_summary(path: str, config: RunnableConfig) -> str:
    """Get a summary of a file: size, line count, language, and for Python files, a list of classes, functions, and imports."""
    start = time.perf_counter()
    root, session = _get_context(config)
    target = root / path
    
    if not target.exists():
        result = f"Error: File '{path}' does not exist."
    elif target.is_dir():
        result = f"Error: '{path}' is a directory."
    else:
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            ext = target.suffix.lower()
            lang_map = {
                ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
                ".tsx": "TSX", ".jsx": "JSX", ".java": "Java", ".go": "Go",
                ".rs": "Rust", ".md": "Markdown", ".json": "JSON",
                ".yaml": "YAML", ".yml": "YAML", ".html": "HTML", ".css": "CSS",
            }
            language = lang_map.get(ext, ext or "unknown")
            info = [
                f"File: {path}",
                f"Language: {language}",
                f"Size: {target.stat().st_size} bytes",
                f"Lines: {len(lines)}",
            ]

            # Python-specific: extract classes, functions, imports
            if ext == ".py":
                try:
                    tree = ast.parse(content)
                    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                    functions = [
                        n.name for n in ast.walk(tree)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    imports = []
                    for n in ast.walk(tree):
                        if isinstance(n, ast.Import):
                            imports.extend(a.name for a in n.names)
                        elif isinstance(n, ast.ImportFrom):
                            imports.append(f"{n.module or ''}")
                    if classes:
                        info.append(f"Classes: {', '.join(classes)}")
                    if functions:
                        info.append(f"Functions: {', '.join(functions)}")
                    if imports:
                        info.append(f"Imports: {', '.join(sorted(set(imports)))}")
                except SyntaxError:
                    info.append("(Could not parse Python AST)")

            result = "\n".join(info)
        except Exception as exc:
            result = f"Error: {exc}"
            
    duration = (time.perf_counter() - start) * 1000
    _log_tool_call(session, "get_file_summary", {"path": path}, result, duration)
    return result

@tool
def save_finding(file_path: str, note: str, config: RunnableConfig, finding_type: str = "observation") -> str:
    """Save an important finding or observation during research. finding_type can be: observation, pattern, issue, dependency, architecture."""
    start = time.perf_counter()
    _, session = _get_context(config)
    
    valid_types = {c.value for c in Finding.FindingType}
    if finding_type not in valid_types:
        finding_type = "observation"

    step_number = ToolCall.objects.filter(session=session).count() + 1
    
    Finding.objects.create(
        session=session,
        file_path=file_path,
        note=note,
        finding_type=finding_type,
        step_number=step_number,
    )
    result = f"Finding saved: [{finding_type}] {file_path} — {note[:100]}"
    
    ToolCall.objects.create(
        session=session,
        tool_name="save_finding",
        tool_input={"file_path": file_path, "note": note, "finding_type": finding_type},
        tool_output=result,
        step_number=step_number,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
    )
    return result

@tool
def get_previous_findings(config: RunnableConfig, repo_url: str = "") -> str:
    """Retrieve findings from previous research sessions on this repository. Helps avoid re-exploring already-known information."""
    start = time.perf_counter()
    _, session = _get_context(config)
    url = repo_url or session.repository.url
    
    past_findings = (
        Finding.objects.filter(session__repository__url=url)
        .exclude(session=session)
        .select_related("session")
        .order_by("-session__created_at", "step_number")[:50]
    )
    
    if not past_findings:
        result = "No previous findings for this repository."
    else:
        lines = []
        for f in past_findings:
            lines.append(
                f"[{f.finding_type}] {f.file_path or 'general'}: {f.note} "
                f"(session {f.session.session_id}, step {f.step_number})"
            )
        result = f"Found {len(lines)} previous finding(s):\n" + "\n".join(lines)
        
    duration = (time.perf_counter() - start) * 1000
    _log_tool_call(
        session, "get_previous_findings",
        {"repo_url": url}, result, duration,
    )
    return result

@tool
def list_past_sessions(config: RunnableConfig, repo_url: str = "") -> str:
    """List past research sessions for this repository, showing their questions and answers."""
    start = time.perf_counter()
    _, session = _get_context(config)
    url = repo_url or session.repository.url
    
    from app.models import ResearchSession as RS
    past = (
        RS.objects.filter(repository__url=url)
        .exclude(pk=session.pk)
        .order_by("-created_at")[:20]
    )
    
    if not past:
        result = "No previous research sessions for this repository."
    else:
        lines = []
        for s in past:
            answer_preview = (s.answer or "")[:200]
            lines.append(
                f"Session {s.session_id} ({s.status}, {s.created_at.isoformat()}):\n"
                f"  Q: {s.question[:150]}\n"
                f"  A: {answer_preview}{'...' if len(s.answer or '') > 200 else ''}\n"
                f"  Tools: {s.total_tool_calls}, Findings: {s.total_findings}, Tokens: {s.total_tokens}"
            )
        result = f"{len(lines)} past session(s):\n\n" + "\n\n".join(lines)
        
    duration = (time.perf_counter() - start) * 1000
    _log_tool_call(
        session, "list_past_sessions",
        {"repo_url": url}, result, duration,
    )
    return result

# The standard list of tools used by the agent
AGENT_TOOLS = [
    list_files,
    read_file,
    search_code,
    get_file_summary,
    save_finding,
    get_previous_findings,
    list_past_sessions,
]
