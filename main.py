import time
import json
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

# Import the existing validated business logic
from swiftbasket_core import hybrid_answer

# Initialize FastAPI application
app = FastAPI(
    title="SwiftBasket Enterprise BI Assistant",
    description="Hybrid SQL + RAG Business Intelligence API"
)

# Add CORS middleware allowing all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's analytical or contextual question")

def is_transient_api_error(e: Exception) -> bool:
    """Detects if an exception is a transient Gemini API error (e.g., 429, 503)."""
    err_str = str(e).lower()
    transient_keywords = ["429", "503", "too many requests", "unavailable", "quota", "overloaded"]
    return any(keyword in err_str for keyword in transient_keywords)

@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

@app.post("/ask")
def ask_question(request: AskRequest):
    """
    Main orchestration endpoint.
    Routes the user's question through the Hybrid SQL + RAG pipeline.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        # First execution attempt
        try:
            result = hybrid_answer(question)
        except Exception as e:
            # Check for transient Gemini API errors to trigger exactly one retry
            if is_transient_api_error(e):
                time.sleep(2)  # Short delay before retry
                result = hybrid_answer(question)
            else:
                raise e  # Non-transient error, escalate to outer block

    except Exception as e:
        if is_transient_api_error(e):
            raise HTTPException(
                status_code=503,
                detail="The AI service is temporarily unavailable. Please try again shortly."
            )
        # Fallback for all other unexpected application errors
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal server error occurred."
        )

    # Map the successful execution payload to a clean JSON response
    response_payload = {
        "question": question,
        "route": result.get("route"),
        "status": result.get("status"),
        "route_reason": result.get("route_reason"),
        "matched_signals": result.get("matched_signals", []),
        "exact_identifier": result.get("exact_identifier"),
        "answer_type": result.get("answer_type"),
        "error": result.get("error")
    }

    if response_payload["route"] == "SQL":
        response_payload["generated_sql"] = result.get("generated_sql")
        
        df = result.get("data")
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Let pandas natively convert NaNs, NaTs, and timestamps to strict JSON (nulls and ISO strings)
            json_str = df.to_json(orient="records", date_format="iso")
            records = json.loads(json_str)
            
            # Use jsonable_encoder to guarantee final serialization safety for FastAPI
            response_payload["data"] = jsonable_encoder(records)
        else:
            response_payload["data"] = []
    else:
        # RAG specific fields
        response_payload["retrieval_status"] = result.get("retrieval_status")
        response_payload["retrieval_method"] = result.get("retrieval_method")
        response_payload["answer"] = result.get("answer")

    return response_payload

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )