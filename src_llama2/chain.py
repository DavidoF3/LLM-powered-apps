"""This module contains functions for loading a ConversationalRetrievalChain"""

import logging

import wandb
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from prompts import load_chat_prompt
import torch

from langchain.prompts import PromptTemplate
import transformers

logger = logging.getLogger(__name__)


def load_vector_store(wandb_run: wandb.run) -> Chroma:
    """Load a vector store from a Weights & Biases artifact
    Args:
        run (wandb.run): An active Weights & Biases run
    Returns:
        Chroma: A chroma vector store object
    """
    # Load artifact of directory containing the vector store
    vector_store_artifact_dir = wandb_run.use_artifact(
        wandb_run.config.vector_store_artifact, type="search_index"
    ).download()
    # Initialise SBERT embeddings model to generate the embedding vectors
    # embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_fn = HuggingFaceEmbeddings(
        model_name=wandb_run.config.embed_model_name,
        model_kwargs= {'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': False}
        )
    # Load vector store
    vector_store = Chroma(
        embedding_function=embedding_fn, persist_directory=vector_store_artifact_dir
    )

    return vector_store


def init_model(wandb_run: wandb.run, hf_auth: str):
    """Load a quantsised llama2 model
    Args:
        model_id (str): ID of model as specified in Huggin Face 
        hf_auth (str): Hugging Face authentication key to use the llama2 model
    Returns:
        model: object of llama2 model
    """
    # Set quantization configuration to load large model with less GPU memory
    # - this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # This configuration object uses the model configuration from Hugging Face 
    # to set different model parameters
    model_config = transformers.AutoConfig.from_pretrained(
        wandb_run.config.llm_model_name,
        use_auth_token=hf_auth
    )

    # Download and initialize the model 
    model = transformers.AutoModelForCausalLM.from_pretrained(
        wandb_run.config.llm_model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth
    )
    
    return model.eval()


def load_chain(wandb_run: wandb.run, vector_store: Chroma, hf_auth: str):
    """Load a ConversationalQA chain from a config and a vector store
    Args:
        wandb_run (wandb.run): An active Weights & Biases run
        vector_store (Chroma): A Chroma vector store object
        model_id (str): ID of model as specified in Huggin Face 
        hf_auth (str): Hugging Face authentication key to use the llama2 model
    Returns:
        ConversationalRetrievalChain: A ConversationalRetrievalChain object
    """

    # Set k as input variable !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # retriever - used to retrieve docs from similarity search
    retriever = vector_store.as_retriever(search_kwargs=dict(k=3))

    # Setup Llama2 LLM model and tokeniser
    # - create objects of llama2 llm model and tokeniser
    model = init_model(wandb_run, hf_auth)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        wandb_run.config.llm_model_name,
        use_auth_token=hf_auth
        )

    pipe = transformers.pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        do_sample=True,           
        temperature=wandb_run.config.chat_temperature,
        repetition_penalty=wandb_run.config.repetition_penalty,   
        max_new_tokens=wandb_run.config.max_new_tokens
        )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Load from json file !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    prompt_template = """<s>[INST] Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:  [/INST]"""

    # In the prompt template we define two inputs: "context", "question"
    prompt_template = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    return llm, prompt_template, retriever

    # # Load prompt artifact
    # chat_prompt_dir = wandb_run.use_artifact(
    #     wandb_run.config.chat_prompt_artifact, type="prompt"
    # ).download()
    # qa_prompt = load_chat_prompt(f"{chat_prompt_dir}/prompt.json")

    # # ConversationalRetrievalChain
    # # - utility defined in LangChain to respond to user questions
    # # - main difference to RetrievalQA is that  with former, can pass in 
    # #   your chat history to the model
    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=retriever,
    #     combine_docs_chain_kwargs={"prompt": qa_prompt},
    #     return_source_documents=True,
    # )
    # return qa_chain


def get_answer(
    llm: HuggingFacePipeline,
    prompt_template: PromptTemplate,
    retriever,
    question: str,
    # chat_history: list[tuple[str, str]],
):
    """Get an answer from a ConversationalRetrievalChain along with user question
    Args:
        chain (ConversationalRetrievalChain): A ConversationalRetrievalChain object
        question (str): The question to ask
        chat_history (list[tuple[str, str]]): A list of tuples of (question, answer)
    Returns:
        str: The answer to the question
    """

    # Fix inputs to function get_answer() !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    docs = retriever.get_relevant_documents(question)
    # The context is a concatenation of the retrieved docs
    context = "\n\n".join([doc.page_content for doc in docs])
    # Populate the prompt with the context and question variables
    prompt = prompt_template.format(context=context, question=question)
    
    result = llm.predict(prompt)

    response = f"Answer:\t{result}"

    return response