import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import pinecone
import os  # Use environment variables for security

# Debugging: Print API key status (DO NOT print actual keys for security)
print("PINECONE_API_KEY is set:", bool(os.getenv("PINECONE_API_KEY")))
print("OPENAI_API_KEY is set:", bool(os.getenv("OPENAI_API_KEY")))

# Initialize FastAPI app
app = FastAPI()

# Load API keys from environment variables (Corrected)
PINECONE_API_KEY = os.getenv("pcsk_2Vxs26_UY18jaV64wX7tmKdf5PgY6KjPKtUQF6oef8HsRQ1Bcj2xfsYbqnpSLaKfnL8noQ")  # Retrieve from environment
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")  # Default to "us-east-1"
OPENAI_API_KEY = os.getenv("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")  # Retrieve from environment

# Validate API keys
if PINECONE_API_KEY is None or OPENAI_API_KEY is None:
    raise ValueError("Missing API keys! Ensure PINECONE_API_KEY and OPENAI_API_KEY are set.")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index("googlesheet")  # Make sure this matches your Pinecone index name

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
        # Generate embedding for the query
        query_vector = openai.Embedding.create(
            input=[request.query], model="text-embedding-ada-002"
        )["data"][0]["embedding"]

        # Search in Pinecone
        search_results = index.query(query_vector, top_k=5, include_metadata=True)

        # Extract context from search results
        if "matches" in search_results and search_results["matches"]:
            context = "\n".join([match["metadata"]["text"] for match in search_results["matches"]])
        else:
            context = "No relevant information found."

        # Generate a response using OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant using Pinecone for knowledge retrieval."},
                {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {request.query}"}
            ]
        )

        return {"response": response["choices"][0]["message"]["content"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  
    uvicorn.run(app, host="0.0.0.0", port=port)
