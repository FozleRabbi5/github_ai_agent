# GitHub AI Agent

## OpenAI setup

Create a local `.env` file in the project root (`github_ai_agent/.env`) before running the app:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4.1
OPENAI_EXTRACTOR_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

You can copy the template from `.env.example`.

The app now loads `.env` automatically from `github_ai_agent/.env` and passes the API key explicitly to both chat and embedding clients.
