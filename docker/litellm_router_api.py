"""OpenAI-compatible FastAPI server with Conduit + LiteLLM routing."""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Conduit + LiteLLM Router", version="1.0.0")

# Get API keys from environment
api_keys = {
    'OpenAI': os.getenv('OPENAI_API_KEY'),
    'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'Google': os.getenv('GOOGLE_API_KEY'),
    'Groq': os.getenv('GROQ_API_KEY'),
}

# Build model list from available keys
# IMPORTANT: Multiple deployments per model allow Conduit to learn and route
model_list = []

if api_keys['OpenAI']:
    # Multiple gpt-4o-mini deployments for Conduit to route between
    model_list.extend([
        {
            'model_name': 'gpt-4o-mini',
            'litellm_params': {'model': 'gpt-4o-mini', 'api_key': api_keys['OpenAI']},
            'model_info': {'id': 'gpt-4o-mini-1'},
        },
        {
            'model_name': 'gpt-4o-mini',
            'litellm_params': {'model': 'gpt-4o-mini', 'api_key': api_keys['OpenAI']},
            'model_info': {'id': 'gpt-4o-mini-2'},
        },
    ])

if api_keys['Anthropic']:
    # Multiple Claude deployments for Conduit to route between
    model_list.extend([
        {
            'model_name': 'claude-3-5-haiku',
            'litellm_params': {'model': 'claude-3-5-haiku-20241022', 'api_key': api_keys['Anthropic']},
            'model_info': {'id': 'claude-3-5-haiku-1'},
        },
        {
            'model_name': 'claude-3-5-haiku',
            'litellm_params': {'model': 'claude-3-5-haiku-20241022', 'api_key': api_keys['Anthropic']},
            'model_info': {'id': 'claude-3-5-haiku-2'},
        },
    ])

if api_keys['Groq']:
    # Multiple Groq model deployments
    model_list.extend([
        {
            'model_name': 'llama-3.1-8b',
            'litellm_params': {'model': 'groq/llama-3.1-8b-instant', 'api_key': api_keys['Groq']},
            'model_info': {'id': 'llama-3.1-8b-1'},
        },
        {
            'model_name': 'llama-3.1-8b',
            'litellm_params': {'model': 'groq/llama-3.1-8b-instant', 'api_key': api_keys['Groq']},
            'model_info': {'id': 'llama-3.1-8b-2'},
        },
    ])

# Initialize router
redis_url = os.getenv('REDIS_URL')
router = Router(model_list=model_list, redis_host=redis_url if redis_url else None)

# Set up Conduit routing strategy
# Note: use_hybrid is removed - Router now always uses hybrid routing
strategy = ConduitRoutingStrategy(cache_enabled=bool(redis_url))
ConduitRoutingStrategy.setup_strategy(router, strategy)
use_hybrid = True  # Always true now

print(f"✅ Conduit + LiteLLM Router initialized")
print(f"   Models: {len(model_list)}")
print(f"   Hybrid Routing: {use_hybrid}")
print(f"   Redis: {'enabled' if redis_url else 'disabled'}")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: list[ChatMessage]
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False


@app.get("/")
async def root():
    return {
        "service": "Conduit + LiteLLM Router",
        "version": "1.0.0",
        "models": len(model_list),
        "routing": "ML-powered (Hybrid: UCB1→LinUCB)" if use_hybrid else "ML-powered (LinUCB)"
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint.

    Shows unique model_names (not deployment IDs) so Conduit can route between deployments.
    """
    # Get unique model names (not deployment IDs)
    unique_models = {}
    for m in model_list:
        model_name = m["model_name"]
        if model_name not in unique_models:
            unique_models[model_name] = {
                "id": model_name,  # Use model_name as ID, not deployment ID
                "object": "model",
                "created": 1677610602,
                "owned_by": m["litellm_params"]["model"].split("/")[0] if "/" in m["litellm_params"]["model"] else "openai"
            }

    return {
        "object": "list",
        "data": list(unique_models.values())
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Route through LiteLLM with Conduit strategy
        response = await router.acompletion(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
        )

        if request.stream:
            async def generate():
                async for chunk in response:
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
