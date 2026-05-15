from langgraph.checkpoint.memory import InMemorySaver


_CHECKPOINTER = InMemorySaver()


def get_checkpointer() -> InMemorySaver:
    return _CHECKPOINTER
