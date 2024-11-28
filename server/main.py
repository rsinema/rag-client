from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from psycopg2 import connect
from psycopg2.extras import RealDictCursor
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

origins = [
    "*",
]

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

# Load the tokenizer and model for generating responses
response_model_name = "facebook/blenderbot-400M-distill"
response_tokenizer = AutoTokenizer.from_pretrained(response_model_name)
response_model = AutoModelForSeq2SeqLM.from_pretrained(response_model_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conn = None
cursor = None

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
    # embed the text using huggingface model
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = embedding_model(**inputs)
    embedded_text = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedded_text

def generate_bot_response(text: str) -> str:
    # generate the response using the huggingface model
    inputs = response_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = response_model.generate(**inputs)
    bot_response = response_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return bot_response

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
def trigger_rag(user_input: str, owner: str):
    print("Running RAG Inference")
    embedded_user_text = embed_text(user_input)
    related_documents = get_related_documents(embedded_text=embedded_user_text, owner=owner)
    # TODO: could do more prompt engineering here
    augmented_text = user_input + " ".join(related_documents)
    rag_bot_response = generate_bot_response(augmented_text)
    return {"bot_response": rag_bot_response}

# The user can just call model inference API to get a direct bot response w/ no RAG
@app.get("/model-inference")
def read_bot_response(user_input: str):
    bot_response = generate_bot_response(user_input)
    return {"bot_response": bot_response}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)