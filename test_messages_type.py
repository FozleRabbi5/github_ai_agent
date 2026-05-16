import json

# Check if there is any ResearchSession in the DB with status 'FAILED' or 'RUNNING'
# and print its tool calls or messages if we can
import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "github_ai_agent.settings")
django.setup()

from app.models import ResearchSession
from langchain_core.messages import ToolMessage

# Let's just create a dummy agent and see what types of messages are in the state
from app.agents.repository_agent import ResearchAgent
session = ResearchSession.objects.last()
if session:
    print(f"Session: {session.session_id}")
