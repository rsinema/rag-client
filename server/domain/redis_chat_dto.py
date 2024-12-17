
class RedisChatDTO:
    def __init__(self, chat_id, message, context=None):
        self.chat_id = chat_id
        self.message = message
        self.context = context