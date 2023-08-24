# Building a baseline LLM App

Here we build a baseline application that exposes a chat interface to W&B users, where they can ask questions to the W&B bot.

# Step 1 - Ingestion

In this step we ingest our documents into a vector store, so that we can retrieve them later as new queries come from our users. 

In the file [ingest.py](/src/ingest.py), we define the functions for ingesting documents.
* `load_documents`: function used to load our documents (stored in Markdown format).
* `chunk_documents`: documents loaded by `load_documents` may be longer than what we can stuff into the prompt. Hence, `chunk_documents` splits long documents into a list of chunks, using a Markdown text splitter.
* `create_vector_store`: receives the list of document chunks, passes them through an embedding model to generate embeddings, and stores the docs and embeddings in a vector store object.

    We could keep the document store and vector embeddings locally, but might need more than that, since: `1)` Might want to run our application on a remote server. So would be helpful to pull the docs and embeddings from a remote location. `2)` Might want to track the `lineage` of our application. Such that for a specific request in production we know which version of our prompts, embedding models, documents was used. Useful to troubleshoot errors in user queries. For this reason:

* `log_dataset`: this function uses W&B artifact to log the dataset containing the list of docs.
* `log_index`: this function uses W&B artifact to log the index with vector store embeddings.
* `log_prompt`: this function uses W&B artifact to log the prompts. As we experiment with different versions of our system and user template, we want to understand how different prompt versions influence our results in production or evaluation.

* `ingest_data`: puts the above functions (`load_documents`,`chunk_documents`,`create_vector_store`) together. It ingests a directory of Markdown files and returns a vector store.
* `get_parser`: this function allows passing a list of arguments, to control the directory or our docs, chunk size, chunk overlap, etc.
* `main`: puts everything together i.e. `get_parser` + `ingest_data` + `log_dataset`, `log_index`, `log_prompt`.

Run with command
```python
python ingest.py --docs_dir "../docs_sample"
```

# Step 2 - Building App

In this step we build a simple web app that allows W&B users to ask questions to the bot.

In the file [config.py](/src/config.py), we define our App configurations. For example, W&B project options, location of vector store and prompt template, LLM model to use, etc.

In the file [chain.py](/src/chain.py), we define the functions:
* `load_vector_store`: function used to load the vector store from W&B artifacts
* `load_chain`: create a LangChain conversationalQA chain object from a config and a vector store
* `get_answer`: generate answer from the conversationalQA chain by passing a user question and chat history

In the file [app.py](/src/app.py), we create a class to define a simple chatbot that uses a `LangChain` chain and `Gradio UI` to answer questions about wandb documentation. This class uses the inputs defined in [config.py](/src/config.py) and the functions from [chain.py](/src/chain.py).

Run with command
```python

# If we want to start tracing LangChain with W&B (run in python)
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

python app.py
```

This App could be deployed, for example, in [HuggingFace Spaces](https://huggingface.co/docs/hub/spaces-sdks-gradio) or in Replit.


# Error analysis

If some of the answers from the LLM are flagged as unhelpful by users, you can use the information logged by W&B to narrow down the causes for bad answers. Examples:
* Wrong document retrieved from vector store from a given user query. So would need to improve something in the docs search.
* Maybe the model did not perform so well and did not find the correct answer within the prompt (containing retrieved documents). Could improve by using better prompt engineering or changing to a more powerful model.