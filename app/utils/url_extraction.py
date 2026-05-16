import logging
import re

logger = logging.getLogger(__name__)

def extract_github_url_from_text(text: str) -> str | None:
    # 1. Regex extraction
    match = re.search(r"(https?://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)", text)
    if match:
        return match.group(1).rstrip('.')
    
    # 2. OpenAI Fallback
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        messages = [
            SystemMessage(content="Extract the GitHub repository URL from the following text. Return ONLY the URL. If no GitHub URL is found, return the exact word 'NONE'."),
            HumanMessage(content=text)
        ]
        response = llm.invoke(messages)
        content = response.content.strip()
        if content != "NONE" and "github.com" in content:
            return content
    except Exception as e:
        logger.error(f"OpenAI extraction failed: {e}")
        
    return None

def clean_question_text(question: str, url: str) -> str:
    cleaned = question.replace(url, "")
    # Remove separators like '-' and extra spaces
    cleaned = re.sub(r'^\s*-\s*', '', cleaned)
    cleaned = re.sub(r'\s*-\s*$', '', cleaned)
    # Strip quotes and extra spaces
    cleaned = cleaned.strip(" '\"- \t\n\r")
    return cleaned
