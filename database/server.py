import os
from service import RAGService
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import uvicorn
import logging

app = FastAPI()
model = None
service = None

@app.on_event("startup")
async def load_model():
    global model, service
    model_name = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info('Loading model...')
    model = SentenceTransformer(model_name)
    service = RAGService(model)
    logger.info('Model loaded.')

@app.get("/retrieve")
def retrieve(query: str):
    return service.query_database(query)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)