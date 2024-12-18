import json
import os
import redis
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import torch


@dataclass
class PromptConfig:
    system_prompt: str
    max_context_length: int = 2048
    response_max_length: int = 256
    temperature: float = 0.9

class RAGProcessor:
    def __init__(
        self,
        model_name: str,
        prompt_config: PromptConfig,
        redis_client: redis.StrictRedis
    ):
        print("Initializing RAG Processor...")
        self.text_generator = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            print("Using CUDA")
            self.text_generator.to('cuda')
        elif torch.backends.mps.is_available():
            print("Using MPS")
            self.text_generator.to('mps')
        
        self.text_generator.eval()
            
        self.prompt_config = prompt_config
        self.redis_client = redis_client

    def format_prompt(self, user_message: str, context_data: Optional[str] = None) -> str:
        """
        Formats the complete prompt with system prompt, context, and user message
        """
        # Start with system prompt
        formatted_prompt = f"### System:\n{self.prompt_config.system_prompt}\n\n"
        
        # Add context if provided
        if context_data:
            formatted_prompt += f"### Context:\n{context_data}\n\n"
        
        # Add user message
        formatted_prompt += f"### User:\n{user_message}\n\n### Assistant:\n"
        
        return formatted_prompt

    def generate_response(self, prompt: str) -> str:
        """
        Generates response using the text generation model
        """
        print("Generating response... (this may take a while)")
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt', max_length=self.prompt_config.max_context_length, truncation=True)

        if torch.cuda.is_available():
            tokenized_prompt = tokenized_prompt.to('cuda')
        elif torch.backends.mps.is_available():
            tokenized_prompt = tokenized_prompt.to('mps')

        # Generate response
        response = self.text_generator.generate(
            **tokenized_prompt,
            max_length=self.prompt_config.max_context_length + self.prompt_config.response_max_length,
            temperature=self.prompt_config.temperature,
        )

        response = self.tokenizer.batch_decode(response, skip_special_tokens=True)
        response = response[0]

        print(f"Response generated successfully:\n {response}")
    
        # Extract just the assistant's first response
        assistant_response = response.split("### Assistant:\n")[1]

        print(f"Assistant response: {assistant_response}")
        
        # Look for common end markers and take everything before them
        end_markers = ["### END", "END", "###", "### User:", "### System:", "---", "Let me know"]
        for marker in end_markers:
            if marker in assistant_response:
                assistant_response = assistant_response.split(marker)[0]
                break
        
        print(f"Processed response: {assistant_response}")
        
        return assistant_response.strip()



    def process_request(self, request_data: Dict) -> str:
        """
        Processes a single request with user message and optional context
        """
        user_message = request_data['message']
        context_data = request_data.get('context')  # Optional context from vector DB
        
        formatted_prompt = self.format_prompt(user_message, context_data)
        # print(f"Formatted prompt: {formatted_prompt}")
        return self.generate_response(formatted_prompt)

    def poll_queue(self, queue_name: str):
        """
        Continuously polls Redis queue for new requests
        """
        while True:
            # Blocking pop from the queue
            print(f"Polling queue: {queue_name}")
            _, request_bytes = self.redis_client.blpop(queue_name)

            
            if request_bytes:
                request_dict = json.loads(request_bytes)
                chat_id = request_dict['chat_id']
                print(f"Received request: {chat_id}")
                
                try:
                    response = self.process_request(request_dict)
                    if not response:
                        response = "I'm sorry, I couldn't find a relevant response."
                    print(f"Generated response: {response}")
                    self.redis_client.set(chat_id, response)
                except Exception as e:
                    error_response = json.dumps({"error": str(e)})
                    print(f"Error processing request: {e}")
                    self.redis_client.set(chat_id, error_response)

def main():
    load_dotenv()
    
    # Load configuration from environment
    queue_name = os.getenv('REDIS_QUEUE', 'chat-queue')
    model_name = os.getenv('MODEL_NAME', 'meta-llama/Llama-3.2-1B-Instruct')
    redis_host = os.getenv('REDIS_HOST', 'redis')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    
    # Define system prompt
    system_prompt = os.getenv('SYSTEM_PROMPT', """You are a helpful AI assistant for me, Riley. I am a 25 year old male that has a Bachelor degree in Computer Science, and I am currently working on a Master degree in Computer Science. Please use the provided context to answer 
    questions accurately and concisely. Focus on techinical aspects of my skills, and please try to make me sound competent and a good recruit. If you're unsure or the context doesn't contain 
    relevant information, say so.""")

    temp = os.getenv('MODEL_TEMPERATURE', 0.9)
    temp = float(temp)
    
    # Initialize components
    prompt_config = PromptConfig(
        system_prompt=system_prompt,
        temperature=temp
    )
    
    redis_client = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        db=0
    )
    
    processor = RAGProcessor(
        model_name=model_name,
        prompt_config=prompt_config,
        redis_client=redis_client
    )
    
    # Start processing queue
    print("Processing requests...")
    processor.poll_queue(queue_name)

if __name__ == "__main__":
    print("Starting RAG Processor...")
    main()