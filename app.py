"""
Sidenote Web — FastAPI 后端
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel

from core import extract_article, process_with_llm, render_html


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("警告：未设置 GEMINI_API_KEY 环境变量")
    yield


app = FastAPI(title="Sidenote", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    url: str


class ProcessResponse(BaseModel):
    html: str
    title: str
    title_zh: str


@app.post("/api/process", response_model=ProcessResponse)
async def process_article(req: ProcessRequest):
    try:
        title, text = extract_article(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        data = process_with_llm(title, text)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    html = render_html(data, req.url)
    return ProcessResponse(
        html=html,
        title=data.get("title", ""),
        title_zh=data.get("title_zh", ""),
    )


# Serve frontend
@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = Path(__file__).parent / "static" / "index.html"
    return index_path.read_text(encoding="utf-8")
