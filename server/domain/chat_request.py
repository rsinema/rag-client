from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    chat_id: str
    timestamp: str