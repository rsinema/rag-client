from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_input: str
    conversation_id: str
    timestamp: str