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