from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from psycopg2 import connect
from psycopg2.extras import RealDictCursor
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
import httpx
from typing import List, Dict
from pydantic import BaseModel


app = FastAPI()

origins = [
    "*",
]

timeout = httpx.Timeout(
    connect=60.0,  # time to establish a connection
    read=60.0,     # time to wait for a response
    write=20.0,    # time to wait for writing the request
    pool=20.0      # time to wait for a connection from the connection pool
)

# LLM_API_URL = "https://100.78.249.37:5002" # Tailscale IP
# LLM_API_URL = "https://172.22.104.182:5002" # Private IP
# LLM_API_URL = "http://localhost:5002" # local host
LLM_API_URL = "http://host.docker.internal:5002"


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conn = None
cursor = None

class Message(BaseModel):
    role: str
    content: str

conversations: Dict[str, List[Message]] = {}

@app.on_event("startup")
def startup_event():
    global conn, cursor
    try:
        conn = connect(
            host=os.getenv("DATABASE_HOST", "db"),
            database=os.getenv("DATABASE_NAME", "mydb"),
            user=os.getenv("DATABASE_USER", "myuser"),
            password=os.getenv("DATABASE_PASSWORD", "mypassword"),
            cursor_factory=RealDictCursor
        )
        if conn:
            print("Connected to the database successfully")
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error connecting to the database: {e}")
    cursor = conn.cursor()

@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down the server")
    if cursor:
        cursor.close()
    if conn:
        conn.close()

def embed_text(text: str) -> list[float]:
    # Send the request to the external LLM server
    response = httpx.post(f"{LLM_API_URL}/encode", json={"text": text}, verify=False, timeout=timeout)
    response.raise_for_status()  # Raise an error for HTTP response codes >= 400
    return response.json()

def generate_bot_response(messages: List[Message]) -> str:
    # Send a request to the `/generate` endpoint
    response = httpx.post(f"{LLM_API_URL}/generate", json={"messages": [m.dict() for m in messages]}, verify=False, timeout=timeout)
    response.raise_for_status()
    return response.json().get("prediction", "")

def get_related_documents(embedded_text: list[float], owner: str) -> list[str]:
    # Returns the related document segments as a list of strings

    # Get the related documents from the database using the embedded input text
    # Option 1: Both the input text and the related documents are embedded with the same model. Easiest to implement but probably not the best results
    cursor.execute("""
        SELECT 
            content, 
            embedding <-> %s::vector AS distance
        FROM 
            documents
        WHERE
            owner = %s
        ORDER BY 
            distance
        LIMIT 5
    """, (embedded_text, owner))
    
    related_documents = cursor.fetchall()
    return [doc["content"] for doc in related_documents]

    # Option 2: Use a dual encoder model (siamese training) for the input text and the related documents to be compared in the same space. By far the hardest to implement but likely the best results
    pass

@app.get("/")
def read_root():
    print("Hello World")
    return {"Hello": "World"}

# The main logic for the RAG system
@app.get("/rag-inference")
def trigger_rag(user_input: str, owner: str, conversation_id: str):
    print("Running RAG Inference")

    # Retrieve or create conversation history
    if conversation_id == "" or conversation_id is None:
        # Generate a new conversation ID
        conversation_id = str(len(conversations) + 1)
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user message to conversation
    conversations[conversation_id].append(Message(role="user", content=user_input))

    embedded_user_text = embed_text(user_input)
    related_documents = get_related_documents(embedded_text=embedded_user_text, owner=owner)

    prompt = "You are a chatbot for a porfolio website. Here is revelant information according to what the user asked: "
    context = prompt + " ".join(related_documents)
    print("Context: ", context)
    system_message = Message(role="system", content=f"Context: {context}")

    messages_for_llm = [system_message] + conversations[conversation_id]
    
    # Add bot response to conversation history
    rag_bot_response = generate_bot_response(messages_for_llm)
    conversations[conversation_id].append(Message(role="assistant", content=rag_bot_response))
    
    return {"bot_response": rag_bot_response, "conversation_id": conversation_id, "owner": owner}

# The user can just call model inference API to get a direct bot response w/ no RAG
@app.get("/model-inference")
def read_bot_response(user_input: str):
    bot_response = generate_bot_response(user_input)
    return {"bot_response": bot_response}

# Endpoint to get conversation history
@app.get("/conversation/{conversation_id}")
def get_conversation(conversation_id: str):
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conversations[conversation_id]}

# Endpoint to clear conversation history
@app.delete("/conversation/{conversation_id}")
def clear_conversation(conversation_id: str):
    if conversation_id in conversations:
        del conversations[conversation_id]
    return {"message": "Conversation cleared"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)