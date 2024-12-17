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