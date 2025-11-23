"""OpenAI-compatible FastAPI server with Conduit + LiteLLM routing."""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from litellm import Router
from conduit_litellm import ConduitRoutingStrategy

app = FastAPI(title="Conduit + LiteLLM Router", version="1.0.0")

# Get API keys from environment
api_keys = {
    'OpenAI': os.getenv('OPENAI_API_KEY'),
    'Anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'Google': os.getenv('GOOGLE_API_KEY'),
    'Groq': os.getenv('GROQ_API_KEY'),
}

# Build model list from available keys
model_list = []

if api_keys['OpenAI']:
    model_list.extend([
        {
            'model_name': 'gpt-4o-mini',
            'litellm_params': {'model': 'gpt-4o-mini', 'api_key': api_keys['OpenAI']},
            'model_info': {'id': 'gpt-4o-mini-openai'},
        },
        {
            'model_name': 'gpt-4o',
            'litellm_params': {'model': 'gpt-4o', 'api_key': api_keys['OpenAI']},
            'model_info': {'id': 'gpt-4o-openai'},
        },
    ])

if api_keys['Anthropic']:
    model_list.extend([
        {
            'model_name': 'claude-3-5-sonnet',
            'litellm_params': {'model': 'claude-3-5-sonnet-20241022', 'api_key': api_keys['Anthropic']},
            'model_info': {'id': 'claude-3-5-sonnet'},
        },
        {
            'model_name': 'claude-3-5-haiku',
            'litellm_params': {'model': 'claude-3-5-haiku-20241022', 'api_key': api_keys['Anthropic']},
            'model_info': {'id': 'claude-3-5-haiku'},
        },
    ])

if api_keys['Google']:
    model_list.append({
        'model_name': 'gemini-1.5-flash',
        'litellm_params': {'model': 'gemini/gemini-1.5-flash', 'api_key': api_keys['Google']},
        'model_info': {'id': 'gemini-1.5-flash'},
    })

if api_keys['Groq']:
    model_list.append({
        'model_name': 'llama-3.1-70b',
        'litellm_params': {'model': 'groq/llama-3.1-70b-versatile', 'api_key': api_keys['Groq']},
        'model_info': {'id': 'llama-3.1-70b-groq'},
    })

# Initialize router
redis_url = os.getenv('REDIS_URL')
router = Router(model_list=model_list, redis_host=redis_url if redis_url else None)

# Set up Conduit routing strategy
use_hybrid = os.getenv('USE_HYBRID_ROUTING', 'true').lower() == 'true'
strategy = ConduitRoutingStrategy(use_hybrid=use_hybrid, cache_enabled=bool(redis_url))
ConduitRoutingStrategy.setup_strategy(router, strategy)

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
    """OpenAI-compatible models endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": m["model_info"]["id"],
                "object": "model",
                "created": 1677610602,
                "owned_by": m["litellm_params"]["model"].split("/")[0] if "/" in m["litellm_params"]["model"] else "openai"
            }
            for m in model_list
        ]
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
