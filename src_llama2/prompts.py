"""Prompts for the chatbot and evaluation."""
import json
import logging
import pathlib
from typing import Union

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)


def load_eval_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        human_template = f_name.open("r").read()
    else:
        logger.warning(
            f"No human prompt provided. Using default human prompt from {__name__}"
        )
        # human_template provides:
        # - question from our evaluation dataset, 
        # - the LLM answer
        # - original answer
        # - request model to provide the grade
        human_template = """\nQUESTION: {query}\nCHATBOT ANSWER: {result}\n
        ORIGINAL ANSWER: {answer} GRADE:"""

    # Here, assuming that we are using a chat-based model for evaluation
    # - so prompt consists of two parts: 
    #    1) system message prompt - tells the LLM its role
    #    2) human message prompt  - 
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """You are an evaluator for the W&B chatbot.You are given a question, the chatbot's answer, and the original answer, 
        and are asked to score the chatbot's answer as either CORRECT or INCORRECT. Note 
        that sometimes, the original answer is not the best answer, and sometimes the chatbot's answer is not the 
        best answer. You are evaluating the chatbot's answer only. Example Format:\nQUESTION: question here\nCHATBOT 
        ANSWER: student's answer here\nORIGINAL ANSWER: original answer here\nGRADE: CORRECT or INCORRECT here\nPlease 
        remember to grade them based on being factually accurate. Begin!"""
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt