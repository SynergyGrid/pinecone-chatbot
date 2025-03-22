from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from pinecone import Pinecone
import os
import logging

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

from fastapi.responses import FileResponse

@app.get("/", response_class=FileResponse)
def serve_ui():
    return FileResponse("index.html")

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        logging.info(f"Received request: {request.query}")

        # Create embedding
        embedding_response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=request.query
        )
        query_vector = embedding_response.data[0].embedding

        # Query Pinecone
        search_results = index.query(
            vector=query_vector,
            top_k=20,
            include_metadata=True
        )

        # Log results
        print("Top Pinecone results:")
        for match in search_results["matches"]:
            print(match["metadata"].get("text", "No text found"))

        # Extract context
        if "matches" in search_results and search_results["matches"]:
            context = "\n".join([match.metadata["text"] for match in search_results["matches"] if "text" in match.metadata])
        else:
            context = "No relevant information found."

        # GPT response
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI assistant using Pinecone for knowledge retrieval."},
                {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {request.query}"}
            ]
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        logging.error(f"Error in /chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
