"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

TEAM = None
PROJECT = "llmapps"
JOB_TYPE = "production"

default_config = SimpleNamespace(
    project=PROJECT,
    entity=TEAM,
    job_type=JOB_TYPE,
    vector_store_artifact="doc93/llmapps/vector_store:latest",
    chat_prompt_artifact="doc93/llmapps/chat_prompt:latest",
    chat_temperature=0.3,
    repetition_penalty=1.1,
    max_new_tokens=500,
    # max_fallback_retries=1,
    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    llm_model_name='meta-llama/Llama-2-7b-chat-hf',
    # eval_model="gpt-3.5-turbo",
    eval_artifact="doc93/llmapps/generated_examples:v0",
)