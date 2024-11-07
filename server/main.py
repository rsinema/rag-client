from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from psycopg2 import connect
from psycopg2.extras import RealDictCursor
import os

app = FastAPI()

origins = [
    "http://localhost:8080",
]

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
    # TODO: embed the text using huggingface model
    # there is a way to cache the model and the encoder
    embedded_text = [0.1, 0.2, 0.3, 0.4, 0.5]
    return embedded_text

def generate_bot_response(text: str) -> str:
    # TODO: generate the response using the huggingface model
    bot_response = "This is a bot response"
    return bot_response

def get_related_documents(embedded_text: list[float]) -> list[str]:
    # Returns the related document segments as a list of strings

    # TODO: get the related documents from the database using the embedded input text
    # Option 1: Both the input text and the related documents are embedded with the same model. Easiest to implement but probably not the best results
    # Option 2: Use a dual encoder model (siamese training) for the input text and the related documents to be compared in the same space. By far the hardest to implement but likely the best results
    pass

@app.get("/")
def read_root():
    print("Hello World")
    return {"Hello": "World"}

# The main logic for the RAG system
@app.get("/rag-inference")
def trigger_rag(user_input: str):
    embedded_user_text = embed_text(user_input)
    related_documents = get_related_documents(embedded_text=embedded_user_text)
    # TODO: could do more prompt engineering here 
    augmented_text = user_input + " ".join(related_documents)
    rag_bot_response = generate_bot_response(augmented_text)
    return {"rag_bot_response": rag_bot_response}

# The user can just call model inference API to get a direct bot response w/ no RAG
@app.get("/model-inference")
def read_bot_response(user_input: str):
    bot_response = generate_bot_response(user_input)
    return {"bot_response": bot_response}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)