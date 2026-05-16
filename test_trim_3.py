from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, trim_messages

messages = [
    SystemMessage("sys"),
    HumanMessage("h1"),
    AIMessage(content="", tool_calls=[{"name": "foo", "args": {}, "id": "1"}, {"name": "bar", "args": {}, "id": "2"}]),
    ToolMessage(content="r1", tool_call_id="1"),
    ToolMessage(content="r2", tool_call_id="2"),
    AIMessage("done")
]

trimmed = trim_messages(
    messages,
    max_tokens=4,
    token_counter=len,
    strategy="last",
    include_system=True,
    allow_partial=False
)
for m in trimmed:
    print(type(m).__name__, getattr(m, 'tool_calls', None) or getattr(m, 'tool_call_id', None))
