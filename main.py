# -*- coding: utf-8 -*-

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Any
import pandas as pd
import json
import os
import logging

from opendataqna import run_pipeline, run_pipeline_stream
from utilities.cache import RedisCache, cache_key

log = logging.getLogger(__name__)

app = FastAPI(title="Open Data QnA API", version="1.0.0")

# Initialize Redis cache
CACHE = RedisCache()


def _ttl_seconds(env_name: str, default: int) -> int:
    """Get TTL from environment variable or use default."""
    try:
        return int(os.getenv(env_name, default))
    except ValueError:
        return default


def _should_bypass_cache(request: Request, body: Optional[dict] = None) -> bool:
    """Check if cache should be bypassed based on headers or body."""
    header_value = request.headers.get("X-Cache-Bypass", "")
    if str(header_value).strip().lower() in {"1", "true", "yes", "y", "on"}:
        return True
    cache_control = request.headers.get("Cache-Control", "")
    if "no-cache" in cache_control.lower():
        return True
    if body:
        payload_value = body.get("cache_bypass")
        if str(payload_value).strip().lower() in {"1", "true", "yes", "y", "on"}:
            return True
    return False


def _cache_log(message: str) -> None:
    """Log cache events if CACHE_LOGGING is enabled."""
    if os.getenv("CACHE_LOGGING", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
        log.info(message)
        print(f"[CACHE] {message}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = ""
    user_question: str
    user_grouping: str
    user_id: Optional[str] = "opendataqna-user@google.com"
    run_debugger: bool = True
    execute_final_sql: bool = True
    cache_bypass: Optional[bool] = False


class ChatResponse(BaseModel):
    session_id: str
    sql: str
    results: List[Any]
    answer: str
    citation: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, raw_request: Request):
    """The main RAG pipeline with Redis caching"""
    try:
        print("request.user_id", request.user_id)
        print("request.session_id", request.session_id)
        
        # Cache setup
        bypass_cache = _should_bypass_cache(raw_request, request.model_dump())
        cache_allowed = CACHE.enabled() and not bypass_cache and bool(request.session_id)
        
        cache_payload = {
            "user_question": request.user_question,
            "user_grouping": request.user_grouping,
            "user_id": request.user_id,
            "run_debugger": request.run_debugger,
            "execute_final_sql": request.execute_final_sql,
        }
        cache_id = cache_key("api_chat", cache_payload)
        
        # Check cache first
        if cache_allowed:
            cached = CACHE.get_json(cache_id)
            if cached is not None:
                _cache_log(f"cache hit: api_chat")
                return cached
        
        # Call your existing async pipeline
        # Note: run_pipeline returns (final_sql, results_df, response_text)
        final_sql, results_df, response_text = await run_pipeline(
            session_id=request.session_id,
            user_question=request.user_question,
            user_grouping=request.user_grouping,
            RUN_DEBUGGER=request.run_debugger,
            EXECUTE_FINAL_SQL=request.execute_final_sql
        )
        
        # Citation placeholder - add your citation logic here if needed
        citation = ""

        if isinstance(results_df, str) and results_df == "Invalid":
            return {
                "session_id": request.session_id,
                "sql": final_sql,
                "results": [],
                "answer": response_text,
                "citation": ""
            }
        
        # Handle DataFrame serialization
        results_list = []
        if isinstance(results_df, pd.DataFrame):
            # Normalize types (decimals, etc) to be JSON serializable
            results_list = results_df.to_dict(orient='records')
        elif isinstance(results_df, str):
            # Sometimes your code returns an error string instead of a DF
            response_text = f"{response_text}\n\nSystem Message: {results_df}"

        response_data = {
            "session_id": request.session_id,
            "sql": final_sql,
            "results": results_list,
            "answer": response_text,
            "citation": str(citation)
        }
        
        # Store in cache
        if cache_allowed and final_sql and not final_sql.startswith("Error"):
            CACHE.set_json(
                cache_id,
                response_data,
                _ttl_seconds("CACHE_TTL_CHAT_SECONDS", 600),
            )
            _cache_log(f"cache set: api_chat")
        
        return response_data
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_endpoint_stream(request: ChatRequest, raw_request: Request):
    """Streaming version of the RAG pipeline - streams the natural language response.
    
    Note: Streaming responses are not cached. Use /api/chat for cached responses.
    """
    
    async def generate_sse():
        try:
            print("request.user_id", request.user_id)
            print("request.session_id", request.session_id)
            
            async for chunk in run_pipeline_stream(
                session_id=request.session_id,
                user_question=request.user_question,
                user_grouping=request.user_grouping,
                RUN_DEBUGGER=request.run_debugger,
                EXECUTE_FINAL_SQL=request.execute_final_sql
            ):
                chunk_type = chunk.get("type")
                
                if chunk_type == "sql":
                    # Send SQL and session_id
                    yield f"data: {json.dumps({'type': 'sql', 'data': chunk['data'], 'session_id': chunk.get('session_id', '')})}\n\n"
                
                elif chunk_type == "results":
                    # Send query results
                    yield f"data: {json.dumps({'type': 'results', 'data': chunk['data']})}\n\n"
                
                elif chunk_type == "text":
                    # Stream response text chunks
                    yield f"data: {json.dumps({'type': 'text', 'data': chunk['data']})}\n\n"
                
                elif chunk_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'data': chunk['data']})}\n\n"
                
                elif chunk_type == "done":
                    yield "data: [DONE]\n\n"
                    
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/cache/health")
async def cache_health():
    """Check if Redis cache is enabled and connected."""
    return {
        "cache_enabled": CACHE.enabled(),
        "cache_type": "redis" if CACHE.enabled() else "disabled",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
