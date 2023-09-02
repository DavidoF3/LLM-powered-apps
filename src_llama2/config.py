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
    prompt_system_artifact="doc93/llmapps/system_prompt_template:latest",
    prompt_user_artifact="doc93/llmapps/user_prompt_template:latest",
    chat_temperature=0.3,
    repetition_penalty=1.1,
    max_new_tokens=500,
    llm_model_name='meta-llama/Llama-2-7b-chat-hf',
    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    # eval_llm_model="meta-llama/Llama-2-7b-chat-hf",
    eval_artifact="doc93/llmapps/generated_examples:v0",
)