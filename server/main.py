from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

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

def embed_text(text: str) -> list[float]:
    # TODO: embed the text using huggingface model
    # there is a way to cache the model and the encoder
    embedded_text = [0.1, 0.2, 0.3, 0.4, 0.5]
    return embedded_text

def generate_bot_response(text: str) -> str:
    # TODO: generate the response using the huggingface model
    bot_response = "This is a bot response"
    return bot_response

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/related-documents")
def read_related_documents(user_input: str):
    embedded_user_text = embed_text(user_input)
    # TODO connect to the database and get the related documents given the user input
    document_text = "This is a document text"
    return {"related-documents": document_text}

@app.get("/model-inference")
def read_bot_response(user_input: str):
    bot_response = generate_bot_response(user_input)
    return {"bot_response": bot_response}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)