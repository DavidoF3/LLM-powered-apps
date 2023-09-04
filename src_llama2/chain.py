"""This module contains functions for loading a ConversationalRetrievalChain"""

import logging

import wandb
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from prompts import load_chat_prompt
import torch

from langchain.callbacks import wandb_tracing_enabled
from langchain.prompts import PromptTemplate
import transformers

from typing import List, Tuple, Optional

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


def create_model_pipeline(wandb_run: wandb.run, hf_auth: str):
    """Create model pipeline using HuggingFacePipeline
    Args:
        wandb_run (wandb.run): An active Weights & Biases run
        hf_auth (str): Hugging Face authentication key to use the llama2 model
    Returns:
        llm: llm pipeline object
    """

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

    return llm


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
    llm = create_model_pipeline(wandb_run, hf_auth)

    # Load system prompt template artifact
    system_prompt_dir = wandb_run.use_artifact(
        wandb_run.config.prompt_system_artifact, type="prompt"
        ).download()
    # Load user prompt template artifact
    user_prompt_dir = wandb_run.use_artifact(
        wandb_run.config.prompt_user_artifact, type="prompt"
        ).download()   

    # Read system prompt template          
    with open(f"{system_prompt_dir}/prompt_system_template.txt", "r") as file:
        system_prompt = file.read()
    # Read user prompt template      
    with open(f"{user_prompt_dir}/prompt_user_template.txt", "r") as file:
        question_template = file.read()

    # Prompt template taking inputs: chat history, user question and context to user question
    prompt_template = """<s>[INST] <<SYS>>\\n""" + system_prompt + """\\n<</SYS>>\\n\\n{PAST_QA} """ + question_template + """[/INST]"""
    # In prompt template - just pass entries that will change from question to question
    prompt_tmp = PromptTemplate(
        template=prompt_template, input_variables=["PAST_QA", "CONTEXT", "QUESTION"]
        )
    
    return llm, prompt_tmp, retriever   

    # OLD - working code -------------------------------------------------------------------------------------------------------------------

    # # Load prompt artifact
    # chat_prompt_dir = wandb_run.use_artifact(
    #     wandb_run.config.chat_prompt_artifact, type="prompt"
    # ).download()
    # qa_prompt = load_chat_prompt(f"{chat_prompt_dir}/prompt.json")


def build_history_prompt(hist):

    if len(hist)>0:
        past_qa=''
        for turn in hist:
            past_qa += f"<s>[INST] {turn[0].strip()} [/INST] {turn[1].strip()} </s>"

        template = '<s>[INST] '
        idx = past_qa.find(template) + len(template)
        return past_qa[idx:] + '<s>[INST]'
    else:
        return ''
    

def get_answer(
    llm: HuggingFacePipeline,
    prompt_template: PromptTemplate,
    retriever,
    question: str,
    chat_history: List[Tuple[str, str]],
    ):
    """Get an answer from a ConversationalRetrievalChain along with user question
    Args:
        llm: llm pipeline to feed questions to
        prompt_template: langchain prompt template to feed to llm
        retriever: chroma object used to retrieve docs from similarity search
        question (str): The question to ask
        chat_history (list[tuple[str, str]]): A list of tuples of (question, answer)
    Returns:
        str: The answer to the question
    """

    # Build past chat history     # Will need to look into hist format -> should be list of [(Q,A),(Q,A),....]      hist avilable in func below!!!!!!!!!!!!!!!!!!!!!!!
    past_qa = build_history_prompt(chat_history)
    
    # Retrieve docs relevant to input question
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Populate prompt template
    prompt = prompt_template.format(PAST_QA=past_qa, CONTEXT=context, QUESTION=question)
    # print('******',prompt)
    # Log LLM results into W&B
    with wandb_tracing_enabled():
        result = llm.predict(prompt)

    response = f"Answer:\t{result}"

    return response