from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from pinecone import Pinecone
import os  # Use environment variables for security

# Debugging: Print API key status (DO NOT print full keys for security)
print("PINECONE_API_KEY is set:", bool(os.getenv("PINECONE_API_KEY")))
print("OPENAI_API_KEY is set:", bool(os.getenv("OPENAI_API_KEY")))

# Print first 5 characters of the keys (for debugging only)
print("PINECONE_API_KEY:", os.getenv("PINECONE_API_KEY")[:5], "***")
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY")[:5], "***")

# Initialize FastAPI app
app = FastAPI()  # ✅ This line was missing!
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domains for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Get from environment
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")  # Default to "us-east-1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Get from environment

# Validate API keys
if not PINECONE_API_KEY.strip() or not OPENAI_API_KEY.strip():
    raise ValueError("Missing API keys! Ensure PINECONE_API_KEY and OPENAI_API_KEY are set.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("googlesheet")  # Make sure this matches your Pinecone index name

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Request model for chat input
class ChatRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "Hello, this is my Pinecone chatbot!"}

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        # Generate embedding for the query (Updated for OpenAI API v1)
        embedding_response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=request.query
        )
        query_vector = embedding_response.data[0].embedding

        # Search in Pinecone
        search_results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        # Extract context from search results
        if "matches" in search_results and search_results["matches"]:
            context = "\n".join([match["metadata"]["text"] for match in search_results["matches"]])
        else:
            context = "No relevant information found."

        # Generate a response using OpenAI
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant using Pinecone for knowledge retrieval."},
                {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {request.query}"}
            ]
        )

        return {"response": response.choices[0].message.content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
