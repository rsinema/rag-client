import os
import uuid
from domain.chat_request import ChatRequest
from domain.chat_response import ChatResponse
from domain.redis_chat_dto import RedisChatDTO
from fastapi import FastAPI
from service.redis_service import RedisService
from dotenv import load_dotenv
import uvicorn
import requests
import json


app = FastAPI()


# {"bot_response": rag_bot_response, "conversation_id": conversation_id, "owner": owner}
def format_context(retrieved_docs):
    # Sort by similarity score in descending order
    sorted_docs = sorted(retrieved_docs, key=lambda x: x['similarity'])
    
    # Extract and combine the text, with clear separation
    formatted_context = []
    for doc in sorted_docs:
        # Add source attribution and the content
        context_piece = f"From {doc['title']}: {doc['text']}"
        formatted_context.append(context_piece)
    
    # Join all pieces with clear separation
    final_context = "\n\n".join(formatted_context)
    
    return final_context

@app.get("/chat", response_model=ChatResponse)  # Better to use response_model parameter
async def chat(rqst: ChatRequest) -> ChatResponse:
    vector_db_url_endpoint = os.getenv("VECTOR_DB_API_URL")

    service = RedisService()
    chat_key = uuid.uuid4().hex
    # retrieve the information from the database and return it
    response = requests.get(f"{vector_db_url_endpoint}/retrieve", params={"query": rqst.message})
    response = response.json()
    chat_dto = RedisChatDTO(chat_id=chat_key, message=rqst.message, context=format_context(response))
    chat_dto_bytes = json.dumps(chat_dto.__dict__).encode('utf-8')
    service.push_queue(chat_dto_bytes)
    response_text = service.poll_db(chat_key, 600)
    response = ChatResponse(response=response_text, timestamp=rqst.timestamp)

    return response

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)