# CS 501R Final Project

## Table of Contents

1. [Introduction](#introduction)
2. [Application Deployment](#application-deployment)
3. [Context Retrieval](#context-retrieval)
4. [Prompting Strategies](#prompting-strategies)
   - [Initial System Prompt](#initial-system-prompt)
   - [Changing the System Prompt](#changing-the-system-prompt)
5. [Future Work](#future-work)
6. [Appendix](#appendix)
   - [Docker Compose](#docker-compose)
   - [Model Code](#model-code)
   - [Server Code](#server-code)
   - [Context Server Code](#context-server-code)

## Introduction

This project is a RAG chatbot application, that has access to information about me so that a user can ask questions about me and my techinical skills. Relevant context from my academic or professional information is retrieved when the user asks a question, and then given to a generative LLM to generate a response to the user's question. The main idea for this is to allow recruiters or potential employers to ask questions about my skills and experience and have the model generate a good answer based on information about me (and maybe be a little impressed by the project).

## Application Deployment

The application is made up of 3 main parts, the server, the context retrieval, and the generative model. The server takes in a chat request, sends a GET request to the context retrieval server to get the context, posts that job to the redis queue, and then polls redis for the response. The model script is constantly polling redis queue for jobs, when it gets one it formats the context and the user input, pushes it through the generative model, and sets the response in redis.

To deploy the application, I have a docker-compose that orchestrates the frontend server, backend server, context server, postgres and redis containers to run together. I wanted to run the model in a container too but it was easier to access the GPU on my machine outside of docker. The API for the backend server is exposed and can be hit, and the frontend is configured to hit the endpoint when the user sends a message in the chat. The containerization of the parts of this application make it so that this is scalable, all it takes in adding more containers. This can be done very easily and effectively using Kubernetes, which would also handle load balancing and container failure. If I was to deploy this publicly that is the route I would take.

The backend and context servers use python with fastAPI to create the API endpoint to hit.

![alt text](RAG-app.drawio.png)

The context retrieval server isn't exactly necessary, but I wanted to split the http functionality and the database retrieval functionality since the retrieval uses a sentence transformer to retrieve the nearest context information.

## Context Retrieval

For this project I created a vector database that contained my academic and professional information from sources like my resume, academic summary, etc. and also wrote up some descriptions of personal projects that I have, descriptions of coursework from relevant classes and descriptions of research projects that I have been working on with Professor Jenkins. The files are stored as plain text, and then chunked, embedded, and stored in the database. I put a server up with an endpoint that will take a text input and return the 5 most similar chunks to the text input based on cosine similarity.

I initially started with smaller chunks around 250-500 characters, but that wasn't enough information to be useful to the generation model. I ended up using chunks with around ~1500 characters, with a 250 character overlap. This gave a large enough chunk to provide a context with a good amount of information in it.

Here is an example of that endpoint getting hit:

```
localhost:5117/retrieve?query=Does Riley have a good understanding of machine learning?

[
    {
        "title": "RileySinemaResume.txt",
        "text": "Riley Sinema (801) 319-9741 ∙ rileysinema@gmail.com ∙ rsinema@byu.edu ∙ linkedin.com/in/rileysinema/ EDUCATION Brigham Young University Provo, UT Master of Science in Computer Science Sept 2024 - Present § Researching calibration methods for probabilistic machine learning Bachelor of Science in Computer Science Sept 2020 – April 2024 § GPA 3.8 § BYU Academic Scholarship § Dean’s List for College of Mathematical and Physical Sciences 2023 LANGUAGES / TECHNOLOGIES Python, Pytorch, Pandas, Numpy, Hugging-face, LLMs, RAG, Computer Vision, C#, .NET, AzureDevOps, SQL, PostgreSQL, NoSQL, Redis, MongoDB, Docker, Kubernetes, Linux, Bash, Git, GitHub, React, TypeScript SKILLS § Proficient understanding machine learning models, algorithms, and application § Proficient understanding of data structures and algorithms § Proficient in design patterns and concepts § Able to work well in teams with other members on projects § Can think creatively about possible project solutions PROJECTS § Self-hosted personal portfolio website built with React TypeScript and dockerized § Researched creating a new metric for more informative evaluation of model calibration § Led a group of students to create a dockerized C# based web app to track personal exercise data § Personal RAG chatbot that queries my personal professional and educational data to answer questions about me EXPERIENCE Marching Order May 2024-Aug 2024 Software Engineer Remote § Implemented critical components of the refactored architecture",
        "similarity": 0.5084527427880312
    },
    {
        "title": "ProbCalResearch.txt",
        "text": "ate. This advancement is particularly significant for high-stakes applications where understanding model confidence on a case-by-case basis is crucial. I am currently expanding the research by investigating the metric's behavior on discrete classification tasks, exploring how the point-wise calibration properties manifest in scenarios with categorical outputs. Technical aspects of my contribution include: Implementation of the novel calibration metric for various neural network architectures Development of experimental frameworks for testing on high-dimensional image datasets Statistical analysis of results comparing our metric against established calibration measures Current investigation of the metric's properties in discrete classification scenarios Documentation and preparation of experimental results for the paper submission This research addresses a fundamental challenge in machine learning reliability assessment, providing practitioners with a more granular and actionable understanding of model confidence.",
        "similarity": 0.6325395545316274
    },
    {
        "title": "CS474.txt",
        "text": "ies of various deep learning architectures through hands-on implementation, reinforcing theoretical concepts with practical engineering challenges.",
        "similarity": 0.6494119376294644
    },
    {
        "title": "ProbCalResearch.txt",
        "text": "I contributed to novel research in machine learning calibration metrics, collaborating on a paper that introduces an innovative approach to evaluating probabilistic neural network confidence. The key innovation of our work is a point-wise calibration metric that enables granular confidence assessment on individual predictions, addressing a significant limitation in existing calibration approaches that typically only provide aggregate measurements. My primary contribution focused on experimental validation, where I designed and executed comprehensive experiments using various image datasets to demonstrate the metric's effectiveness in high-dimensional spaces. The experimental framework involved testing the metric against both standard convolutional neural networks and modern vision transformer architectures, validating its applicability across different model architectures. I implemented efficient data processing pipelines to handle large-scale image datasets, ensuring reproducible results while maintaining computational efficiency. The research extends beyond traditional calibration methods by providing a mathematically rigorous framework for assessing model confidence at the individual prediction level, rather than just in aggregate. This advancement is particularly significant for high-stakes applications where understanding model confidence on a case-by-case basis is crucial. I am currently expanding the research by investigating the metric's behavior on discrete classific",
        "similarity": 0.6635307073593115
    },
    {
        "title": "CS474.txt",
        "text": "I completed a comprehensive Deep Learning course that covered multiple domains of neural network architectures and applications, with an emphasis on from-scratch implementation of each model type. The course structure required deep understanding of the mathematical foundations and architectural details of each model, as we were tasked with building implementations without relying on high-level APIs. Key projects and implementations included: Text Generation Transformer: Developed a character-level transformer model implementing self-attention mechanisms, positional encodings, and masked training for autoregressive text generation. The implementation required careful consideration of memory efficiency and gradient flow through the attention layers. Proximal Policy Optimization (PPO) for Reinforcement Learning: Built a complete PPO algorithm implementation, including policy and value networks, advantage estimation, and the clipped objective function. This project required careful implementation of the policy gradient algorithm and proper handling of the exploration-exploitation trade-off. U-Net Architecture for Image Segmentation: Implemented the full U-Net architecture with skip connections, managing the contracting and expanding paths while maintaining feature map alignment. This project demonstrated understanding of conv-deconv architectures and their applications in dense prediction tasks. Generative Adversarial Network (GAN): Created a GAN implementation for MNIST digit g",
        "similarity": 0.6817864369281263
    }
]
```

There are a couple chunks that are shorter that sometimes get returned and don't have any close relation to the input query. In the example above the third chunk shows, that

## Prompting Stratagies

I constructed a simple prompt that included some information context like this:

```
### System:\n<A statement about the LLM's role, how it should respond, etc.>\n\n
### Context:\n<Information that is retrieved from the database. Snippets from my resume, project descriptions, coursework descriptions, etc. >\n\n
### User:\n<The question from the user>\n\n
### Assistant:\n
```

The model would then generate from the perspective of the assistant. A lot of prompt engineering went into the system section of the prompt to try and get the model to produce output that I thought was decent and followed the theme of the application (making me seem like I am smart and would be a good engineer).

Here is my initial system prompt:

```
"""
You are a helpful AI assistant for me, Riley. I am a 25 year old male. Please use the provided context to answer questions accurately and concisely. Focus on techinical aspects of my skills, and please try to make me sound competent and a good recruit. If you're unsure or the context doesn't contain relevant information, say so. When you are finished, say '### END'.
"""
```

Using this prompt I was able to get relatively good answers. Here is an example of a question from a user and the model's response to it:

```
User Question:
Does Riley understand Deep Learning?

Model Response:
Yes, Riley understands Deep Learning. He completed a comprehensive Deep Learning course that covered multiple domains of neural network architectures and applications, with an emphasis on from-scratch implementation of each model type. His courses reinforced theoretical concepts with practical engineering challenges, and he was awarded the Dean’s List for his academic performance. Riley also collaborated on a research paper introducing an innovative point-wise calibration metric, which provides granular confidence assessment on individual predictions. Overall, his understanding of deep learning principles is excellent.
```

I thought that response was pretty good (it did make me sound pretty competent). It read from my description of CS 474 about deep learning, read from my resume about being on the Dean's list, and talked about research work that I am doing with Professor Jenkins.
Here is another example:

```
User Question:
Does Riley know how to use C# well?

Model Response:
Yes, Riley Sinema is proficient in using C#. According to the information provided in the resume, Riley works as a Software Engineer and contributes to a significant modernization effort of a mission-critical web application that manages graduation ceremonies for educational institutions. The resume mentions that he \"Implemented critical components of the refactored architecture using modern design patterns\" and \"Created responsive UI components using Razor Pages.\" These indicate his ability to utilize modern programming techniques such as C# and its related libraries, which suggests he can be proficient in using C# well. However, without specific documentation or evidence of detailed C# coding, it's not accurate to conclude this with certainty.
```

This response also pretty good, it read from my description of my summer internship about refactoring the company's product, and cited descriptions that I used in my resume about the work I did. At the end, since no actual code was provided it couldn't make a conclusion about my ability to code in C#.

For these examples I was using `Qwen/Qwen2.5-0.5B-Instruct` for my generative model.

Here is a image where the API is hit by using the React web application:
![alt text](chat-rag.png)
This example used `meta-llama/Llama-3.2-1B-Instruct`.

### Changing the System Prompt

I tried a different system prompt that was more in depth, and changed the role of the model and added some more specific instructions for it.

```
"""
You are an AI career advisor specializing in software engineering and machine learning. When answering questions about me, Riley, focus on highlighting:
1. Technical expertise in ML/AI, particularly with LLMs, RAG systems, and NLP
2. Full-stack development capabilities, emphasizing modern technologies and frameworks
3. Academic achievements that demonstrate both theoretical knowledge and practical application
4. Professional accomplishments that show real-world impact

Keep responses focused and technically precise. When discussing skills, provide specific examples from the context that demonstrate mastery. Frame responses to position Riley as a person competent to take on any technical role as a machine learning or software engineer based on the context given.

If the context doesn't contain relevant information for a query, acknowledge this and stay within the bounds of the provided information. Maintain a professional, confident tone while being factually accurate in order to help Riley appear the same.
"""
```

This prompt seemed to get slightly better responses, with more technically focused answers. In the example below, the model pulled information from the context that it was given and mentioned specific examples that I had in my resume to answer the user's question about me. This prompted seemed to be more on topic overall, which makes sense since the prompt was more in depth and specific about how to frame responses to the user.

Here is the example of a response after changing to the new prompt:

```
User Question:
Does Riley understand deep learning well?

Model Response:
Riley appears to have a solid understanding of deep learning concepts, including neural networks, architectures, and applications. He has worked on various projects, such as text generation, image segmentation, and generative adversarial networks, demonstrating his ability to apply deep learning techniques to real-world problems. His experience with LLMs, RAG systems, and NLP is also evident, as highlighted in his projects. Riley's technical expertise in machine learning and software engineering, particularly in Python, PyTorch, and RAG systems, further supports his understanding of deep learning concepts.
```

Here is another example with the UI:
![alt text](chatrag-new-prompt.png)

This prompt made the model a lot more consistent with how it output responses to the user. I did have a time where it seemed to break:
![alt text](chatrag-broke.png)
It output some of my resume at the top, and made it sound like I didn't actually know that much about machine learning (which may or may not be true but I don't want the model saying that to people). I think that increasing the amount of context I give to the model could be beneficial in making it more consistent as well, and when I get a machine with enough compute to do that quickly then I would like to try that.

After seeing the model output like that, I wanted to test it out on a out of domain question:
![alt text](chatrag-ood.png)
I thought that this was pretty funny, but also showed that the model was pretty good about staying on topic about technical skills and connected that to developing machine learning systems for animal care.

## Future Work

This project was very enjoyable to work on, and it helped me to put together a bunch of NLP skills that we learned during the semester. Getting this deployed to a API was also great experience, because that is a big difference from running it in a jupyter notebook. I would love to get this published publicly in some way, I just need to figure out somewhere to deploy and host my model that won't be too expensive.

I would also like to make the chat persistent, currently it is completely stateless and doesn't push any previous messages through the model. I would need to store the chats somehow, and add previous messages into the model's prompt. This would make the chatting more like a actual chatbot, and less like a simple question-answer responding.

## Appendix

### Docker Compose

```yaml
services:
  postgres:
    image: ankane/pgvector:latest
    container_name: rag_db
    environment:
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag_db
    ports:
      - "6012:5432"
    volumes:
      - rag_data:/var/lib/postgresql/data

  redis:
    image: redis:6.0
    container_name: rag_redis
    ports:
      - "6013:6379"

  frontend:
    build:
      context: ./client
      dockerfile: Dockerfile
    container_name: rag_frontend
    ports:
      - "5115:80"

  backend:
    build:
      context: ./server
      dockerfile: Dockerfile
    container_name: rag_backend
    ports:
      - "5116:8000"
    depends_on:
      - postgres
      - redis
      - context-server
    env_file:
      - ./server/.env

  context-server:
    build:
      context: ./database
      dockerfile: Dockerfile
    container_name: rag_context_server
    ports:
      - "5117:8000"
    env_file:
      - ./database/.env

volumes:
  rag_data:
```

### Model code

```python
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
```

### Server code

```python
import os
import uuid
from domain.chat_request import ChatRequest
from domain.chat_response import ChatResponse
from domain.redis_chat_dto import RedisChatDTO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from service.redis_service import RedisService
from dotenv import load_dotenv
import uvicorn
import requests
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/chat", response_model=ChatResponse)  # Better to use response_model parameter
async def chat(rqst: ChatRequest) -> ChatResponse:
    vector_db_url_endpoint = os.getenv("VECTOR_DB_API_URL")

    service = RedisService()
    chat_key = uuid.uuid4().hex
    # retrieve the information from the database and return it
    response = requests.get(f"{vector_db_url_endpoint}/retrieve", params={"query": rqst.user_input})
    response = response.json()
    chat_dto = RedisChatDTO(chat_id=chat_key, message=rqst.user_input, context=format_context(response))
    chat_dto_bytes = json.dumps(chat_dto.__dict__).encode('utf-8')
    service.push_queue(chat_dto_bytes)
    response_text = service.poll_db(chat_key, 600)
    response = ChatResponse(response=response_text, timestamp=rqst.timestamp)

    return response

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Context server code

server.py

```python
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
```

service.py

```python
import os
import pandas as pd
from repo import fast_pg_insert, query_similar_chunks
from utils import pdf2txt
from sentence_transformers import SentenceTransformer
import click

MODEL_NAME = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2')
CHUNK_LENGTH = int(os.getenv('CHUNK_LENGTH', '500'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))

class RAGService:
    def __init__(self, model: SentenceTransformer = None):
        if model:
            self.model = model
        else:
            print('Loading model...')
            self.model = SentenceTransformer(MODEL_NAME)
            print('Model loaded.')

    def embed_query(self, query: str):
        '''
        Embed a query using the SentenceTransformer model.

        Parameters:
        query (str): The query to embed.
        '''
        return self.model.encode([query])[0].tolist()

    def _chunk_text(self, text, n=500, overlap=50):
        '''
            Split a text into overlapping chunks of length n.

            Parameters:
            text (str): The text to split into chunks.
            n (int): The length of each chunk.
            overlap (int): The number of characters to overlap between chunks.

            Returns:
            List[str]: A list of overlapping text chunks.
        '''
        return [text[i:i+n] for i in range(0, len(text), n-overlap)]

    def _process_doc(self, file_path):
        '''
            Process a document by loading it from disk, splitting it into chunks, and processing each chunk.

            Parameters:
            file_path (str): The path to the file to process.

            Returns:
            List[str]: A list of processed text chunks.
        '''

        text = ''
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = self._chunk_text(text, CHUNK_LENGTH, CHUNK_OVERLAP)
        chunks = [' '.join(chunk.split()) for chunk in chunks]
        return chunks

    def _embed_doc(self, file_path, model, verbose=False):
        '''
            Embed a document by loading it from disk, splitting it into chunks, and embedding each chunk.

            Parameters:
            file_path (str): The path to the file to embed.
            model (SentenceTransformer): The sentence transformer model to use for embedding.

            Returns:
            Tuple[List[str], List[np.array]]: A tuple containing a list of text chunks and a list of chunk embeddings.
        '''
        if verbose:
            print(f'Embedding {file_path}...')
        chunks = self._process_doc(file_path)
        return chunks, model.encode(chunks)

    def _prepare_doc_for_db(self, file_path, model, verbose=False):
        '''
            Prepare a document for database insertion by embedding it and creating a DataFrame.

            Parameters:
            file_path (str): The path to the file to process.
            model (SentenceTransformer): The sentence transformer model to use for embedding.

            Returns:
            pd.DataFrame: A DataFrame containing the embeddings and associated metadata.
        '''
        chunks, embeddings = self._embed_doc(file_path, model, verbose)
        title = os.path.basename(file_path)

        chunk_offsets = []
        curr_offset = 0
        for chunk in chunks:
            chunk_offsets.append(curr_offset)
            curr_offset += (len(chunk) + 1 - CHUNK_OVERLAP)


        embeddings_list = [embedding.tolist() for embedding in embeddings]

        data = {
            'doc_title': [title] * len(chunks),
            'chunk_text': chunks,
            'chunk_number': [int(x) for x in range(1, len(chunks) + 1)],
            'begin_offset': chunk_offsets,
            'embedding': embeddings_list
        }
        return pd.DataFrame(data)

    def insert_doc_to_db(self, file_path, columns=['doc_title', 'chunk_text', 'chunk_number', 'begin_offset', 'embedding'], verbose=False):
        '''
            Process a document, prepare it for database insertion, and insert it into the database.

            Parameters:
            file_path (str): The path to the file to process.
            model (SentenceTransformer): The sentence transformer model to use for embedding.
            columns (List[str]): A list of column names in the target table that correspond to the DataFrame columns.

            Returns:
            None
        '''
        if verbose:
            print('Loading model...')
        model = SentenceTransformer(MODEL_NAME)
        if verbose:
            print("Begining insertion process...")
        if file_path.endswith('.pdf'):
            if verbose:
                print("Converting PDF to text...")
            pdf2txt(file_path)
            txt_file_path = file_path.replace('.pdf', '.txt')
            file_path = txt_file_path
        elif not file_path.endswith('.txt'):
            print('Unsupported file type. Please provide a PDF or TXT file.')
            return
        df = self._prepare_doc_for_db(file_path, model, verbose)
        if verbose:
            print("Inserting chunks...")
        fast_pg_insert(df, columns)

    def query_database(self, query, n=5, verbose=False, books=False, extended=False):
        '''
            Query the database for documents containing the given text.

            Parameters:
            query (str): The text to search for in the database.

            Returns:
            List[Tuple[str, str]]: A list of tuples containing the document title and the matching text.
        '''
        if verbose:
            print('Embedding query...')
        query_embedding = self.model.encode([query])[0].tolist()
        if verbose:
            print('Querying database...')

        if extended:
            print('Extended not implemented yet.')
            # chunk_results = query_similar_chunks(query_embedding, n)
            # results_dict = []
            # curr_title = ''
            # for result in chunk_results:
            #     if result[0] != curr_title:
            #         curr_title = result[0]
            #     # book_text = get_book_text_by_title(curr_title)
            #     offset = result[3]

            #     start = max(0, offset - CHUNK_LENGTH)
            #     end = min(len(book_text), offset + CHUNK_LENGTH)

            #     results_dict.append({'title': result[0], 'text': book_text[start:end], 'similarity': result[2]})
            results = query_similar_chunks(query_embedding, n)
            results_dict = [{'title': result[0], 'text': result[1], 'similarity': result[2]} for result in results]
        else:
            results = query_similar_chunks(query_embedding, n)
            results_dict = [{'title': result[0], 'text': result[1], 'similarity': result[2]} for result in results]

        return results_dict

@click.command()
@click.argument('file_path')
@click.option('--verbose', is_flag=True, help='Enable verbose mode')
def add_document(file_path, verbose):
    '''
    Add a document to the database.

    Parameters:
    file_path (str): The path to the file to add.
    '''
    service = RAGService()
    service.insert_doc_to_db(file_path, verbose=verbose)

@click.command()
@click.argument('dir_path')
@click.option('--verbose', is_flag=True, help='Enable verbose mode')
def add_docs(dir_path, verbose):
    '''
    Add all documents in a directory to the database.

    Parameters:
    dir_path (str): The path to the directory containing the files to add.
    '''
    service = RAGService()
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            service.insert_doc_to_db(file_path, verbose=verbose)

@click.command()
@click.argument('query')
@click.option('--n', default=5, help='Number of results to return')
@click.option('--verbose', is_flag=True, help='Enable verbose mode')
def query(query, n, verbose):
    '''
    Query the database for documents containing the given text.

    Parameters:
    query (str): The text to search for in the database.
    n (int): The number of results to return.
    '''
    service = RAGService()
    results = service.query_database(query, n, verbose)
    for result in results:
        print(f'Title: {result["title"]}')
        print(f'Text: {result["text"]}')
        print(f'Similarity: {result["similarity"]}')
        print()

@click.group()
def cli():
    pass

cli.add_command(add_document)
cli.add_command(add_docs)
cli.add_command(query)

if __name__ == '__main__':
    cli()
```

repo.py

```python
import io
import os
import pandas as pd
import psycopg2

from typing import List

from dotenv import load_dotenv
import click

load_dotenv()

CONNECTION_STRING = os.getenv('CONNECTION_STRING', "postgresql://rag_user:rag@localhost:6012/rag_db")
EMBEDDING_LENGTH = os.getenv('EMBEDDING_LENGTH', 384)

CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

INITIALIZE_INFO_EMBEDDINGS_TABLE = f'''
                CREATE TABLE IF NOT EXISTS info_embeddings (
                    id SERIAL PRIMARY KEY,
                    doc_title TEXT,
                    chunk_text TEXT,
                    chunk_number INTEGER,
                    begin_offset INTEGER,
                    embedding vector({EMBEDDING_LENGTH})
                );
                '''

DROP_INFO_EMBEDDINGS = "DROP TABLE IF EXISTS info_embeddings;"

CREATE_INDEX = '''
                CREATE INDEX IF NOT EXISTS embedding_idx ON info_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                '''

REMOVE_INDEX = "DROP INDEX IF EXISTS embedding_idx;"

INSERT_DOC = '''
             INSERT INTO info_embeddings (doc_title, chunk_text, chunk_number, begin_offset, embedding)
             VALUES (%s, %s, %s, %s, %s);
             '''

QUERY_SIMILAR_CHUNKS = '''
                        SELECT doc_title, chunk_text, embedding <=> %s::vector AS distance, begin_offset
                        FROM info_embeddings
                        ORDER BY distance
                        LIMIT %s;
                        '''

CLEAR_ALL_EMBEDDINGS = "DELETE FROM info_embeddings;"

def initialize_info_embeddings_table():
    '''
        Create the PostgreSQL table for storing book embeddings.

        Parameters:
        None

        Returns:
        None
    '''

    with psycopg2.connect(CONNECTION_STRING) as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_EXTENSION)
            cursor.execute(INITIALIZE_INFO_EMBEDDINGS_TABLE)
            connection.commit()

def drop_info_embeddings():
    '''
        Drop the PostgreSQL table for storing book embeddings.

        Parameters:
        None

        Returns:
        None
    '''

    with psycopg2.connect(CONNECTION_STRING) as connection:
        with connection.cursor() as cursor:
            cursor.execute(DROP_INFO_EMBEDDINGS)
            connection.commit()

def create_index():
    '''
        Create the PostgreSQL index for the book embeddings.

        Parameters:
        None

        Returns:
        None
    '''

    with psycopg2.connect(CONNECTION_STRING) as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_INDEX)
            connection.commit()

def remove_index():
    '''
        Remove the PostgreSQL index for the book embeddings.

        Parameters:
        None

        Returns:
        None
    '''

    with psycopg2.connect(CONNECTION_STRING) as connection:
        with connection.cursor() as cursor:
            cursor.execute(REMOVE_INDEX)
            connection.commit()

def clear_embeddings():
    '''
        Clear the PostgreSQL table of all data.

        Parameters:
        None

        Returns:
        None
    '''

    with psycopg2.connect(CONNECTION_STRING) as connection:
        with connection.cursor() as cursor:
            cursor.execute(CLEAR_ALL_EMBEDDINGS)
            connection.commit()

def insert_chunk(book_title, chunk_text, chunk_number, begin_offset, embedding):
    '''
        Insert a document chunk into the PostgreSQL database.

        Parameters:
        book_title (str): The title of the book.
        chunk_text (str): The text of the chunk.
        chunk_number (int): The chunk number within the chapter.
        embedding (np.array): The embedding of the chunk.

        Returns:
        None
    '''
    # Convert types before insertion
    chunk_number = int(chunk_number)  # Convert np.int64 to Python int


    with psycopg2.connect(CONNECTION_STRING) as connection:
        with connection.cursor() as cursor:
            try:
                cursor.execute(INSERT_DOC, (book_title, chunk_text, chunk_number, begin_offset, embedding))
                connection.commit()
                print(f"Successfully inserted chunk {chunk_number}")
            except Exception as e:
                print(f"Error inserting chunk: {e}")
                print(f"Types: book_title={type(book_title)}, chunk_text={type(chunk_text)}, "
                      f"chunk_number={type(chunk_number)}, embedding={type(embedding)}")
                connection.rollback()
                raise

def fast_pg_insert(df: pd.DataFrame, columns: List[str]) -> None:
    """
        Inserts data from a pandas DataFrame into a PostgreSQL table using the COPY command for fast insertion.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be inserted.
        connection (str): The connection string to the PostgreSQL database.
        table_name (str): The name of the target table in the PostgreSQL database.
        columns (List[str]): A list of column names in the target table that correspond to the DataFrame columns.

        Returns:
        None
    """
    conn = psycopg2.connect(CONNECTION_STRING)
    _buffer = io.StringIO()
    df.to_csv(
        _buffer,
        sep='\t',          # Use tab as separator instead of semicolon
        index=False,
        header=False,
        escapechar='\\',   # Add escape character
        doublequote=True,  # Handle quotes properly
        na_rep='\\N'       # Proper NULL handling
    )
    _buffer.seek(0)
    with conn.cursor() as c:
        c.copy_from(
                file=_buffer,
                table='info_embeddings',
                sep='\t',              # Match the separator used in to_csv
                columns=columns,
                null='\\N'            # Match the null representation
            )
    conn.commit()
    conn.close()

def query_similar_chunks(embedding, top_n=5):
    '''
        Query the PostgreSQL database for similar embeddings.

        Parameters:
        embedding (np.array): The embedding to query for.
        top_n (int): The number of similar embeddings to return.

        Returns:
        List: A list of similar embeddings.
    '''

    with psycopg2.connect(CONNECTION_STRING) as connection:
        with connection.cursor() as cursor:
            cursor.execute(QUERY_SIMILAR_CHUNKS, (embedding, top_n))
            results = cursor.fetchall()

    return results

@click.group()
def cli():
    pass

@cli.command()
def init_db():
    """Initialize the database and create the embeddings table."""
    print("Initializing database...")
    print(f"Connection string: {CONNECTION_STRING}")
    print(f"Embedding length: {EMBEDDING_LENGTH}")
    initialize_info_embeddings_table()
    click.echo("Database initialized and index created.")

@cli.command()
def drop_db():
    """Drop the embeddings table."""
    drop_info_embeddings()
    click.echo("Embeddings table dropped.")

@cli.command()
def clear_db():
    """Clear all embeddings from the table."""
    clear_embeddings()
    click.echo("All embeddings cleared from the table.")

@cli.command()
def remove_idx():
    """Remove the index from the embeddings table."""
    remove_index()
    click.echo("Index removed from the embeddings table.")

@cli.command()
def create_idx():
    """Remove the index from the embeddings table."""
    create_index()
    click.echo("Index removed from the embeddings table.")

if __name__ == '__main__':
    cli()
```
