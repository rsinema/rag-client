import os
import time
import redis
from dotenv import load_dotenv

class RedisService:
    '''
    RedisService is a class that provides a simple interface to interact with a Redis database.
    '''

    def __init__(self):
        load_dotenv()
        host = os.getenv('REDIS_HOST', 'localhost')
        port = os.getenv('REDIS_PORT', 6379)
        db = os.getenv('REDIS_DB', 0)
        self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)
        self.queue_key = os.getenv('REDIS_QUEUE', 'chat-queue')

    def set_value(self, key, value):
        try:
            self.client.set(key, value)
            return True
        except Exception as e:
            print(f"Error setting value: {e}")
            return False

    def get_value(self, key):
        try:
            return self.client.get(key)
        except Exception as e:
            print(f"Error getting value: {e}")
            return None

    def delete_value(self, key):
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            print(f"Error deleting value: {e}")
            return False
        
    def poll_db(self, key, timeout=60):
        while timeout > 0:
            value = self.get_value(key)
            if value:
                self.delete_value(key)
                return value
            timeout -= 1
            time.sleep(1)
        return None
    
    def push_queue(self, value):
        try:
            self.client.rpush(self.queue_key, value)
            return True
        except Exception as e:
            print(f"Error pushing to queue: {e}")
            return False