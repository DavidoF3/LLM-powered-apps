"""Evaluate a ConversationalRetrievalChain on a dataset of questions and answers."""
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import wandb
from chain import load_vector_store, load_chain, get_answer
from config import default_config
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from prompts import load_eval_prompt
from tqdm import tqdm

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import transformers


def load_eval_dataset(config: SimpleNamespace) -> pd.DataFrame:
    """Load a dataset of questions and answers from a Weights & Biases artifact
       generated
    Args:
        config (SimpleNamespace): A config object
    Returns:
        pd.DataFrame: A dataframe of questions and answers
    """
    # Load data from a wandb Table artifact - to make sure we track the lineage
    # of the dataset as we do the evaluation
    artifact = wandb.use_artifact(config.eval_artifact)
    # Download artifact
    artifact_dir = Path(artifact.download())
    # Load data
    eval_dataset = pd.read_csv(artifact_dir / "generated_examples.csv")
    return eval_dataset


def generate_answers(
    eval_dataset: pd.DataFrame, 
    llm: HuggingFacePipeline,
    prompt_template: PromptTemplate, 
    retriever 
) -> pd.DataFrame:
    """Generate answers for a dataset of questions and answers
    Args:
        eval_dataset (pd.DataFrame): A dataframe of questions and answers
        llm: llm pipeline to feed questions to
        prompt_template: langchain prompt template to feed to llm
        retriever: chroma object used to retrieve docs from similarity search
    Returns:
        pd.DataFrame: A dataframe of questions, answers, and model answers
    """
    answers = []
    # Run QA chain on each question and append answers
    for query in tqdm(eval_dataset["question"], total=len(eval_dataset)):
        # print('//////',query)
        try:
            result = get_answer(llm,prompt_template,retriever,question=query,chat_history=[])
            answers.append(result)
        except:
            answers.append('nan')        
        # print('-----',result)

    # Make new column in pandas dataframe
    eval_dataset["model_answer"] = answers
    # Convert dataframe to csv
    eval_dataset.to_csv("eval_with_answers.csv", index=False)

    return eval_dataset


def evaluate_answers(
    eval_dataset: pd.DataFrame, config: SimpleNamespace, wandb_run: wandb.run, hf_auth: str
) -> pd.DataFrame:
    """Evaluate a dataset of questions, ideal answers, and model-generated answers
    Args:
        eval_dataset (pd.DataFrame): A dataframe of questions, answers, and model answers
        config (SimpleNamespace): A config object
    Returns:
        pd.DataFrame: A dataframe of questions, answers, model answers, and model scores
    """

    # Object of function to generate prompt for evaluation
    eval_prompt = load_eval_prompt()

    # Create object of LLM model used to evaluate answer of LLM to be used in app 

    # llm = ChatOpenAI(
    #     model_name=config.eval_model,
    #     temperature=0,
    # )

    # QAEvalChain from LangChain - used to evaluate model generated answers
    eval_chain = QAEvalChain.from_llm(llm, prompt=eval_prompt)

    # Process the evaluation dataset
    examples = []
    predictions = []
    for i in range(len(eval_dataset)):
        examples.append(
            {
                "query": eval_dataset["question"].iloc[i],
                "answer": eval_dataset["answer"].iloc[i],
            }
        )
        predictions.append(
            {
                "query": eval_dataset["question"].iloc[i],
                "answer": eval_dataset["answer"].iloc[i],
                "result": eval_dataset["model_answer"].iloc[i],
            }
        )
    # Pass processed evaluation dataset into evaluation chain
    graded_outputs = eval_chain.evaluate(examples, predictions)
    # Save results in new dataframe column
    eval_dataset["model_score"] = [x.get("text", "None") for x in graded_outputs]
    return eval_dataset


def log_results(eval_dataset: pd.DataFrame) -> None:
    """Once processed evaluation, log evaluation results to a Weights & Biases Artifact
    Args:
        eval_dataset (pd.DataFrame): A dataframe of questions, answers, model answers, and model scores
    """
    # Calculate LLM accuracy = no correct answers / length of dataset
    model_accuracy = len(eval_dataset[eval_dataset["model_score"] == "CORRECT"]) / len(
        eval_dataset
    )
    # Log accuracy to W&B
    wandb.log({"model_accuracy": model_accuracy})
    # Log eval dataset csv to W&B artifact
    eval_dataset.to_csv("eval_results.csv", index=False)
    artifact = wandb.Artifact("eval_results", type="eval_results")
    artifact.add_file("eval_results.csv")
    wandb.log_artifact(artifact)
    # Log eval dataset as W&B table for further exploration
    wandb.log({"eval_results": wandb.Table(dataframe=eval_dataset)})


if __name__ == "__main__":
    # Run evaluation under new W&B run
    with wandb.init(project=default_config.project, config=default_config, job_type="eval") as run:
        eval_dataset = load_eval_dataset(default_config)
        # vector_store = load_vector_store(run, os.environ["OPENAI_API_KEY"])
        # qa_chain = load_chain(run, vector_store, os.environ["OPENAI_API_KEY"])
        # eval_dataset = generate_answers(eval_dataset, qa_chain)
        # eval_dataset = evaluate_answers(eval_dataset, default_config)
        vector_store = load_vector_store(run)
        llm, prompt_tmp, retriever  = load_chain(run, vector_store, os.environ["HF_API_KEY"])
        eval_dataset = generate_answers(eval_dataset, llm, prompt_tmp, retriever )
        eval_dataset = evaluate_answers(eval_dataset, default_config, run, os.environ["HF_API_KEY"])
        log_results(eval_dataset)