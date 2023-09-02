"""Ingest a directory of documentation files into a vector store and store the relevant artifacts in Weights & Biases"""
import argparse
import json
import logging
import os
import pathlib
from typing import List, Tuple

import langchain
import wandb
from langchain.cache import SQLiteCache
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredMarkdownLoader
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma

langchain.llm_cache = SQLiteCache(database_path="langchain.db")

logger = logging.getLogger(__name__)


def load_documents(data_dir: str) -> List[Document]:
    """Load documents from a directory of markdown files

    Args:
        data_dir (str): The directory containing the markdown files

    Returns:
        List[Document]: A list of documents
    """
    # Create list of paths to Markdown files
    md_files = list(map(str, pathlib.Path(data_dir).glob("*.md")))
    # Process files using the LangChain utility UnstructuredMarkdownLoader
    documents = [
        UnstructuredMarkdownLoader(file_path=file_path).load()[0]
        for file_path in md_files
    ]
    return documents


def chunk_documents(
    documents: List[Document], chunk_size: int = 500, chunk_overlap=0
) -> List[Document]:
    """Split documents into chunks

    Args:
        documents (List[Document]): A list of documents to split into chunks
        chunk_size (int, optional): The size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The number of tokens to overlap between chunks. Defaults to 0.

    Returns:
        List[Document]: A list of chunked documents.
    """
    # Split documents into chunks
    markdown_text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_documents = markdown_text_splitter.split_documents(documents)
    return split_documents


def create_vector_store(
    documents,
    vector_store_path: str = "./vector_store",
) -> Chroma:
    """Create a ChromaDB vector store from a list of documents

    Args:
        documents (_type_): A list of documents to embed and add to the vector store
        vector_store_path (str, optional): The path to the vector store. Defaults to "./vector_store".

    Returns:
        Chroma: A ChromaDB vector store object containing the documents along with their embeddings.
    """

    # Initialise SBERT embeddings model to generate the embedding vectors
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
        )
    # Use Chroma to pass the docs and embedding_function, to store 
    # the docs and their embeddings to a certain directory location 
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=vector_store_path,
    )
    vector_store.persist()
    return vector_store


def log_dataset(documents: List[Document], run: "wandb.run"):
    """Log a dataset to wandb

    Args:
        documents (List[Document]): A list of documents to log to a wandb artifact
        run (wandb.run): The wandb run to log the artifact to.
    """
    document_artifact = wandb.Artifact(name="documentation_dataset", type="dataset")
    with document_artifact.new_file("documents.json") as f:
        for document in documents:
            f.write(document.json() + "\n")

    run.log_artifact(document_artifact)


def log_index(vector_store_dir: str, run: "wandb.run"):
    """Log a vector store to wandb

    Args:
        vector_store_dir (str): The directory containing the vector store to log
        run (wandb.run): The wandb run to log the artifact to.
    """
    index_artifact = wandb.Artifact(name="vector_store", type="search_index")
    index_artifact.add_dir(vector_store_dir)
    run.log_artifact(index_artifact)


def log_prompt(prompt_system_file: pathlib.Path, 
               prompt_user_file: pathlib.Path, 
               run: "wandb.run"):
    """Log a system and user prompt to wandb

    Args:
        prompt_system_file (pathlib.Path): The system prompt template to log
        prompt_user_file (pathlib.Path): The user prompt template to log
        run (wandb.run): The wandb run to log the artifact to.
    """
    system_prompt_artifact = wandb.Artifact(name="system_prompt_template", type="prompt")
    system_prompt_artifact.add_file(local_path=prompt_system_file)
    run.log_artifact(system_prompt_artifact)

    user_prompt_artifact = wandb.Artifact(name="user_prompt_template", type="prompt")
    user_prompt_artifact.add_file(local_path=prompt_user_file)
    run.log_artifact(user_prompt_artifact)


def ingest_data(
    docs_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    vector_store_path: str,
) -> Tuple[List[Document], Chroma]:
    """Ingest a directory of markdown files into a vector store

    Args:
        docs_dir (str):
        chunk_size (int):
        chunk_overlap (int):
        vector_store_path (str):


    """
    # load the documents
    documents = load_documents(docs_dir)
    # split the documents into chunks
    split_documents = chunk_documents(documents, chunk_size, chunk_overlap)
    # create document embeddings and store them in a vector store
    vector_store = create_vector_store(split_documents, vector_store_path)
    return split_documents, vector_store


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        required=True,
        help="The directory containing the wandb documentation",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="The number of tokens to include in each document chunk",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=0,
        help="The number of tokens to overlap between document chunks",
    )
    parser.add_argument(
        "--vector_store",
        type=str,
        default="./vector_store",
        help="The directory to save or load the Chroma db to/from",
    )
    parser.add_argument(
        "--prompt_system_file",
        type=pathlib.Path,
        default="./prompt_system_template.txt",
        help="The path to the system prompt template to use",
    )
    parser.add_argument(
        "--prompt_user_file",
        type=pathlib.Path,
        default="./prompt_user_template.txt",
        help="The path to the user prompt template to use",
    )
    parser.add_argument(
        "--wandb_project",
        default="llmapps",
        type=str,
        help="The wandb project to use for storing artifacts",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    # config in wandb.init() - used to save inputs to the wandb run, like hyperparameters for 
    #                          a model or settings for a data preprocessing job
    run = wandb.init(project=args.wandb_project, config=args)
    # Ingest docs and return splitted docs (of max specified length) and vectorstore (storing
    # the docs and their embeddings)
    documents, vector_store = ingest_data(
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        vector_store_path=args.vector_store,
    )
    # Log artifacts to wandb
    log_dataset(documents, run)
    log_index(args.vector_store, run)
    log_prompt(args.prompt_system_file, args.prompt_user_file, run)
    # Finish wandb run
    run.finish()


if __name__ == "__main__":
    main()
