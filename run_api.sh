set -a
source .env
set +a
export LLM_PROVIDER=api
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_API_MODEL=openai/gpt-oss-120b
