# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import time

app = FastAPI()

# مسیر مدل GGUF
MODEL_PATH = "./Mistral-7B-Instruct-v0.3.IQ1_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7


@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Local LLM API is running"}

@app.post("/v1/chat/completions")
def chat_completion(req: ChatRequest):
    # متن کامل prompt بر اساس پیام‌ها
    prompt = ""
    for msg in req.messages:
        prompt += f"{msg.role.upper()}: {msg.content}\n"
    prompt += "ASSISTANT:"

    start = time.time()
    output = llm(
        prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=["USER:", "ASSISTANT:"],
    )
    end = time.time()

    text = output["choices"][0]["text"].strip()

    # فرمت OpenAI API
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "usage": {
            "prompt_tokens": output["usage"]["prompt_tokens"],
            "completion_tokens": output["usage"]["completion_tokens"],
            "total_tokens": output["usage"]["total_tokens"],
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "index": 0,
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
