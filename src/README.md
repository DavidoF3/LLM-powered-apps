Here we build a baseline application that exposes a chat interface to W&B users, where they can ask questions to the W&B bot.

* Step 1
  * Ingest our documents into a vector store, so that we can retrieve them later as new queries come from our users. 
  * In the file [ingest.py](/src/ingest.py), we define the functions for ingesting documents.
    * `load_documents`: function used to load our documents (stored in Markdown format).
    * `chunk_documents`: documents loaded by `load_documents` may be longer than what we can stuff into the prompt. Hence, `chunk_documents` splits long documents into a list of chunks, using a Markdown text splitter.
    * `create_vector_store`: receives the list of document chunks, passes them through an embedding model to generate embeddings, and stores the docs and embeddings in a vector store object.

        We could keep the document store and vector embeddings locally, but might need more than that, since: `1)` Might want to run our application on a remote server. So would be helpful to pull the docs and embeddings from a remote location. `2)` Might want to track the `lineage` of our application. Such that for a specific request in production we know which version of our prompts, embedding models, documents was used. Useful to troubleshoot errors in user queries. For this reason:

    * `log_dataset`: this function uses W&B artifact to log the dataset containing the list of docs.
    * `log_index`: this function uses W&B artifact to log the index with vector store embeddings.
    * `log_prompt`: this function uses W&B artifact to log the prompts. As we experiment with different versions of our system and user template, we want to understand how different prompt versions influence our results in production or evaluation.
  
    * `ingest_data`: puts the above functions (`load_documents`,`chunk_documents`,`create_vector_store`) together. It ingests a directory of Markdown files and returns a vector store.
    * `get_parser`: this function allows passing a list of arguments, to control the directory or our docs, chunk size, chunk overlap, etc.
    * `main`: puts everything together i.e. `ingest_data` + `log_dataset`, `log_index`, `log_prompt`.

    * Run with command
        ```python
        python ingest.py --docs_dir "../docs_sample"
        ```