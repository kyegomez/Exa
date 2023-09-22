import os
from typing import List

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi_pagination import PaginationParams, paginate
from fastapi_pagination.bases import AbstractPage
from fastapi_pagination.ext.tortoise import paginate
from loguru import logger
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.limiter import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

class Prompt(BaseModel):
    text: str
    max_length: int = 20

class Deploy:
    def __init__(
        self,
        llm,
        app: FastAPI = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.llm = llm
        self.app = app if app else FastAPI()
        self.host = host
        self.port = port

        @self.app.on_event("startup")
        def load_model():
            self.load_model()

        @self.app.post("/generate/")
        @limiter.limit("5/minute")  # rate limit
        async def generate(prompt: Prompt):
            return self.generate(prompt)

        @self.app.get("/items/")
        async def read_items(params: PaginationParams = Depends()):
            items = [{"item": "item1"}, {"item": "item2"}, {"item": "item3"}]  # example items
            return paginate(items)

        @self.app.get("/download/")
        async def download_file():
            return FileResponse('path_to_file', filename='filename')

    def load_model(self):
        global inference
        inference = self.llm

    async def generate(self, prompt: Prompt):
        try:
            return {"generated_text": inference.run(prompt.text, prompt.max_length)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)

# if __name__ == "__main__":
#     llm = Inference(model_id="gpt2-small")  # replace with your LLM
#     deploy = Deploy(llm)
#     deploy.run()