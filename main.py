from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from pinecone import Pinecone
import os
import logging
from fastapi.responses import FileResponse

# Set up logging
logging.basicConfig(level=logging.INFO)

# Debugging: Check API key status
logging.info(f"PINECONE_API_KEY is set: {bool(os.getenv('PINECONE_API_KEY'))}")
logging.info(f"OPENAI_API_KEY is set: {bool(os.getenv('OPENAI_API_KEY'))}")
logging.info(f"PINECONE_API_KEY: {os.getenv('PINECONE_API_KEY')[:5]} ***")
logging.info(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')[:5]} ***")

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY.strip() or not OPENAI_API_KEY.strip():
    raise ValueError("Missing API keys!")

# Initialize Pinecone and OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("googlesheet")
openai.api_key = OPENAI_API_KEY

class ChatRequest(BaseModel):
    query: str

@app.get("/", response_class=FileResponse)
def serve_ui():
    return FileResponse("index.html")

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        logging.info(f"Received request: {request.query}")

        # Create embedding using text-embedding-3-small
        embedding_response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        )
        query_vector = embedding_response.data[0].embedding

        # Query Pinecone with top_k=50
        search_results = index.query(
            vector=query_vector,
            top_k=50,
            include_metadata=True
        )

        # Filter results by score
        matches = [
            match for match in search_results.get("matches", [])
            if match.get("score", 0) >= 0.75 and "text" in match.metadata
        ]

        # Build context from filtered matches
        context = "\n\n".join(match.metadata["text"] for match in matches) or "No relevant information found."

        # GPT response
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant using Pinecone for knowledge retrieval. "
                        "Always provide clear, well-structured responses. Use section headers, bullet points, and tables whenever helpful."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nUser Query: {request.query}"
                }
            ]
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-index")
def debug_index():
    try:
        results = index.query(vector=[0.0]*1536, top_k=20, include_metadata=True, namespace="")
        output = []
        for i, match in enumerate(results.get("matches", [])):
            text = match["metadata"].get("text", "❌ MISSING TEXT")
            source = match["metadata"].get("source", "no source")
            output.append(f"{i+1}. Source: {source}\n{text}\n{'-'*40}")
        return {"results": output}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
