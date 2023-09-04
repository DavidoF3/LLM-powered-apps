"""A Simple chatbot that uses the LangChain and Gradio UI to answer questions about wandb documentation."""
import os
from types import SimpleNamespace

import gradio as gr
import wandb
from chain import get_answer, load_chain, load_vector_store
from config import default_config

from typing import List, Tuple, Optional


class Chat:
    """A chatbot interface (using a Gradio UI, exposed as a web app) that persists the vector store 
    and chain between calls.
    """

    def __init__(
        self,
        config: SimpleNamespace,
    ):
        """Initialize the chatbot.
        Args:
            config (SimpleNamespace): The configuration.
        """

        # Use configuration in config.py to setup wandb run
        self.config = config
        self.wandb_run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            job_type=self.config.job_type,
            config=self.config,
        )
        self.vector_store = None
        self.llm = None

    def __call__(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        hf_api_key: str = None,
    ):
        """Answer a question about wandb documentation using the LangChain QA chain and vector store retriever.
        Args:
            question (str): The question to answer.
            history (list[tuple[str, str]] | None, optional): The chat history. Defaults to None.
            hf_api_key (str, optional): The Hugging Face access token. Defaults to None.
        Returns:
            list[tuple[str, str]], list[tuple[str, str]]: The chat history before and after the question is answered.
        """
        if hf_api_key is not None:
            hf_key = hf_api_key
        elif os.environ["LLAMA2_API_KEY"]:
            hf_key = os.environ["LLAMA2_API_KEY"]
        else:
            raise ValueError(
                "Please provide your Hugging Face API key as an argument or set the LLAMA2_API_KEY environment variable"
            )

        # When we call the chat interface, what happens in the background is that, in case the vector store
        # or chain (i.e. llm, prompt_template, retriever) are not loaded, we load them below and save them to the chat object
        if self.vector_store is None:
            self.vector_store = load_vector_store(
                wandb_run=self.wandb_run
            )
        if self.llm is None:
            self.llm, self.prompt_template, self.retriever = load_chain(
                self.wandb_run, self.vector_store, hf_auth=hf_key
            )

        history = history or []
        # Make all characters in string lower case (dont need? !!!!!!!!!!!!!!!!!!!!!!!!!!!)
        # question = question.lower()
        # Use get_answer to create a response and append it to the history
        # response = get_answer(
        #     chain=self.chain,
        #     question=question,
        #     chat_history=history,
        # )
        response = get_answer(
            llm=self.llm,
            prompt_template=self.prompt_template,
            retriever=self.retriever,
            question=question,
            chat_history=history,
        )
    
        history.append((question, response))
        return history, history

# Use Gradio Blocks to define the UI
with gr.Blocks() as demo:
    gr.HTML(
        """<div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            Llama2 Wandb QandA Bot
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Hi, I'm a wandb documentaion Q and A bot powered by Llama2.<br>
        Start by typing in your: 1) <a href="https://huggingface.co/settings/tokens" target="_blank">Hugging Face access token</a>, and 2) questions/issues you have related to wandb usage and then press enter.<br>
        Built using <a href="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf" target="_blank">Hugging Face</a>, <a href="https://langchain.readthedocs.io/en/latest/" target="_blank">LangChain</a> and <a href="https://github.com/gradio-app/gradio" target="_blank">Gradio Github repo</a>
        </p>
    </div>"""
    )
    with gr.Row():
        # Textbox to specify Hugging Face access token
        hf_api_key = gr.Textbox(
            type="password",
            label="1) Enter your Hugging Face access token here",
        )
        # Text box for the user question
        question = gr.Textbox(
            label="2) Type in your questions about wandb here and press Enter!",
            placeholder="How do I log images with wandb ?",
        )
    # Store the state
    state = gr.State()
    # Chatbot object
    chatbot = gr.Chatbot()
    # Whenever a user submits a question,
    # - input 1 - we create an object of the Chat class (ny passing the default config)
    # - input 2 - we call the __call__ from the Chat object (need to pass the question, state (history), and hf api key)
    # - input 3 - output of __call__ method (i.e. answer), that can be exposed in the UI (i.e. the chatbot box)
    question.submit(
        Chat(config=default_config,),
        [question, state, hf_api_key],
        [chatbot, state],
    )

if __name__ == "__main__":
    # demo - defined above
    demo.queue().launch(
        share=False, server_name="0.0.0.0", server_port=8884, show_error=True
    )