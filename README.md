# LLM-powered-apps

- [LLM-powered-apps](#llm-powered-apps)
- [Notebooks](#notebooks)
- [Theory of LLMs](#theory-of-llms)
  - [Predicting the next token.](#predicting-the-next-token)
  - [Sampling methods](#sampling-methods)
  - [Training](#training)
  - [Prompting Structure](#prompting-structure)
- [APIs are not enough - App architecture](#apis-are-not-enough---app-architecture)
  - [Embeddings](#embeddings)
  - [Vector databases / Embedding stores](#vector-databases--embedding-stores)
  - [Parsing Documents](#parsing-documents)
- [Evaluating LLM Apps](#evaluating-llm-apps)
  - [Code implementation](#code-implementation)
- [Identifying areas of improvement](#identifying-areas-of-improvement)
- [Strategies and ideas to enhance App](#strategies-and-ideas-to-enhance-app)
  - [Document search](#document-search)
  - [Chains](#chains)
  - [LLMs](#llms)
  - [Prompt engineering](#prompt-engineering)
- [Controlling LLM outputs](#controlling-llm-outputs)
  - [Problems](#problems)
  - [Solution](#solution)
  - [Controlling LLMs with Guardrails AI](#controlling-llms-with-guardrails-ai)
- [Safety considerations](#safety-considerations)
- [Installation of libraries](#installation-of-libraries)

# Notebooks

The notebooks in this repository are related to the topics covered in the section [Theory of LLMs](#theory-of-llms):
* [0_using_APIs_llama2.ipynb](/notebooks/0_using_APIs_llama2.ipynb): covers [tokenisation](#predicting-the-next-token), calling an LLM from an API, and [sampling methods](#sampling-methods).
* [1_generation_llama2.ipynb](/notebooks/1_generation_llama2.ipynb): covers [prompt engineering](#prompting-structure) (zero shot, few shot, adding context & response, and Level 5 prompting) for building a chatbot.
* [2_retreival_qa_llama2.ipynb](/notebooks/2_retreival_qa_llama2.ipynb): covers 
  
# Theory of LLMs

For details about the transformer architecture of LLMs, click the [link](https://jalammar.github.io/illustrated-gpt2/).

## Predicting the next token. 

Basic process:
1) Start with some input text.
2) Split the text into tokens (represented by numbers).
3) Feed this chain of tokens into LLM.
4) LLM returns distribution of probabilities over the entire vocabulary (i.e. all the available tokens to our model), for the next token to continue the sequence.
5) We can select (*sample*) the token with the highest probability for example (i.e. greedy decoding).
6) Append selected token to the input sequence and we repeat the process (i.e. tokanize input sequence, feed to LLM, output vocab probabilities and select next the token).


## Sampling methods

LLMs generate text through sampling. They produce a set of probabilities across the vocabulary, and pick a toke to follow the input sequence. Different methods of sampling are:
* **Greedy decoding** 
  * Pick token with highest probability.
* **Beam search** 
  * Generate multiple candidate sequences to maximise probability of a sequence of tokens. 
  * Often leads to repetitions and lack of meaning.
* **Sampling with temperature**
  * Adjusting temp. affects token probabilities. 
    * Higher values result in more diverse outputs, since tokens with lower probabilities get sampled. 
    * Lower temps makes less probably tokens even less likely.
    * As temp -> 0, it becomes more like **Greedy decoding**.
* **Top p sampling** 
  * Only consider tokes with probabilities above a threshold. Hence, excludes tokens with low probabilities. 
  * Results in high quality generated text. 


## Training

Training of LLMs consists of three main steps ([more details](https://arxiv.org/pdf/2203.02155.pdf)):
1) Pre-training
   * Model learns from a massive dataset. Examples sources used: CommonCrawl, C4, Github, Wikipedia, Books, ArXiv, StackExchange.
   * In this phase, the model can be good at predicting text found in the pre-train dataset.
2) Supervised instruction tuning
   * Model further trained on expert-generated question-answer paths. 
   * Helps to align model with user expectations and follow instructions.
3) Reinforcement learning from human feedback
   * Model trained to optimise for higher quality answers preferred by human judges.
   * Done by outputing several answers with a model and using a labeler (human) to rank the outputs (best to worst). This is used to train a reward model. This reward model is then used to reward for outputs of the LLM, and update its policy using Proximal Policy Optimization (PPO).


## Prompting Structure

The way we prompt can greatly affect the output of the LLM.

There are different prompts maturity levels (prompt taxometry `TELeR`):
* Level 0 - No directive
* Level 1 - Simple one sentence instruction
  
* ... Add more data, details & directives
  
* Level 5 - Complex directive with:
   * Description of high-level goal 
   * Detailed bulleted list of sub-tasks 
   * An explicit statement asking the LLM to explain its output
   * Guidelines on how LLM output will be evaluated
   * And few short examples


# APIs are not enough - App architecture

Problems with LLMs as knowledge bases:
* Trained up to a specific `knowledge cutoff` (knowledge gained after the cutoff not available to the model).
* We don't know what LLMs are trained on (`uncurated training data`). Data might be inaccurate, contain outdated code...
* `Hallucinations` as models sample from probability distributions (non-deterministic outputs i.e. not the same always).
* Expensive to `retrain` (update), so usually use them as is.

Solutions:
* One solution.. add relevant knowledge in the prompt along with the question. But if we have the knowledge why do we need the LLM?
* Define an app archtecture
  * `Docs` serves as our knowledge base. It should be comprehensive and contain answers to most questions (but may be hard to navigate for users).
  * Feed all `docs` into a `document store`.
  * Use an `embedding model` to find relevant documents for user questions.     
    * `Embedding model` converts `docs` into numeric representations and stores them in a `vector database`.
    * Pass user `questions` to the `embedding model` to find similar documents through similarity search.
  * A `prompt template` helps us create a dynamic prompt - that includes the user `question` and similar `docs`.
  * This `prompt` is passed to the `LLM`, for generating an `answer` that is displayed to the user.

<!-- Image  -->
![Image of app architecture with vector database](/images/app_architecture.png)

The implementation of this solution is detailed [here](/README_llm_app.md).

## Embeddings

* Embeddings help find relevant docs without relying on keyword matches alone. Embedding search relies on `semantic overlap` (eg. between a question and different docs).
* By training embedding models on large `question-answer pairs datasets` (eg. Stack Exchange), embeddings can effectively map questions and docs into a `numeric space`, where we can compute `similarity scores`.
  
* Similarity and distance metrics
  * Eucledian distance
    * Length of a line between two points in the vector space.
  * Cosine similarity
    * Angle between two vectors.
  
* Problems with embeddings
  * Generic embeddings (via API) may not capture the meaning of domain-specific docs (eg. chemistry, biology).
  * New terms, proper names, .. may not be well represented.
  
* Solutions 
  * Train your own embedding model
  * Combine with classic search algorithms (keywords, TF-IDF, BM25, ...)


## Vector databases / Embedding stores

[Chroma](https://www.trychroma.com/) is an open-source embeddings store. 

An embedding store - is a software package that abstracts away (simplifies) a lot of the operations covered in the [embeddings section](#embeddings). Its basic functioning:
* First take a set of `docs` that represent the knowledge base.
* Use the embedding store to `embed` them using its `embedding function`.
* When a `query` comes in (from user), the same `embedding function` gets called, it generates an embedding (`query embedding`).
* Embedding stores performs the `nearest neighbor search`, and returns the relevant docs to the LLM context window.

When to use an embedding store?
* When data is sufficiently large (~10k+ embeddings)
  * computing distances to each embedding for each query is too slow or expensive (specially under some distance functions).
* When development velocity is important
  * LLM powered applications need to support many users across many indexes, need to handle data and scale automatically. Convenient that it it "just works".

Speed behind vector databases / embedding store is due to `approximate nearest neighbor (ANN)`:
* `Exact nearest neighbor` requires to calculate distance to every stored embedding for every query (takes `O(N)` operations).
* `ANN` algorithms exploit the structure of the data, and take only `O(log(N))` operations. 
  * Trade `recall` for `speed`. This trade-off can be turned depending on algorithm used.
  * Common methods are: inverted file indexes, locality-sensitive hashing, HNSW.
  * `HNSW` (`graph-based` algorithm) works well for LLM in the loop applications - since data will mutate online quite frequently, and we can iteratively construct and iterate on the graph.

Questions remain:
* Which embedding model is best for my task? 
* How should I chunk up my data to ensure good results?
* Are the neighbors actually relevant? What do I do if they are not?
* How do I incorporate human feedback?


## Parsing Documents

Some docs in our knowledge base can be quite long. Even with LLMs taking large token input windows (GPT-4: 32k, or Anthropic Cloud: 100k), managing long docs efficiently and economically can be challenging. 

Solutions:
* Chunk docs using a [sliding window](https://python.langchain.com/docs/modules/data_connection/document_transformers/) . 
* `Overlapping` chunks to ensure we don't loose useful information (eg. cutting in half sentences or code snippets). 
* Use semantic structure of docs (eg. markdown headers to guide division of docs into chunks).
* Might need to deal with different document types (eg. domain-specific text, unique formats like latex, code, pdfs).
  * [LangChain](https://python.langchain.com/docs/modules/data_connection/document_loaders/) library might be useful - since contains various utility functions to handle different document types.


<!-- ## Assembling components

Check this [link](https://github.com/wandb/wandbot) for wandbot integration into Slack and Discord integrations. -->


# Evaluating LLM Apps
Before starting to improve an LLM app, it's essential to decide how to evaluate the effectiveness of the changes made. This is challenging due to unstructured and stochastic nature of the outputs.

* `Vibes check`: 
  * During LLM training manually prompt and assess the outputs. This is a quick and subjective way of assessing model improvements. But not ideal to objectively compare tens or hundreds of experiments. 

* `Model-based evaluation`: 
  * To automate the evaluation process, we can use another LLM. 
  * Here, we gather a dataset of questions and corresponding ideal answers. 
    * Ideally, the questions come from a production service and ideal answers are manually annotated by experts.
    * We can also create a synthetic dataset (as done in [1_generation_llama2.ipynb](/notebooks/1_generation_llama2.ipynb)).
  * Fit this dataset to the LLM (to be evaluated) and generate the answers.
  * The evaluation LLM then judges if the generated answers are correct, which can then be summarised as an overall accuracy metric.

    ![LLM model-based evaluation](/images/evaluation_model-based.png)

* `Unit testing`:
  * For a fixed query, we extract a set of specific items to validate (test) the query. 
  * Each query example might have a different set of tests items. Aim is to pass as many as possible.
  * This approach requires more effort but can yield more robust results.

* `A/B testing in production`:
  * Collecting user feedback and tracking performance in production. 
    * For example, enable thumbs up and down reactions.
    * Can then monitor the percentage of positive and negative reactions over time.
  * Critical to be able to track performance to a specific version of the LLM (using W&B artifacts).

## Code implementation

The [eval.py](/src_llama2/eval.py) contains functions to run evaluation of the LLM used for the developed app. The evaluation approach followed is `Model-based evaluation`. The functions within [eval.py](/src_llama2/eval.py) are:
* `load_eval_dataset`: load a dataset of questions and answers from a Weights & Biases artifact generated.
* `generate_answers`: generate answers for a dataset of questions and answers. This function takes a `qa_chain` as input, created by the function `load_chain` (in [load_chain.py](/src_llama2/chain.py)).
* `evaluate_answers`: evaluate a dataset of questions, ideal answers, and model-generated answers, by using an evaluation LLM.
* `log_results`: once processed evaluation, log evaluation results to a W&B Artifact.


# Identifying areas of improvement

Start by looking at the App architecture, and considering which elements might be under-performing. 

![Image of app architecture with vector database](/images/app_architecture.png)

Error analysis (finding at which point models break) can be conducted with tools like `W&B Tracer`. Taking the image above as a reference:
* We don't have much control on the quality of `users questions`. But can improve the quality of the UI and guiding users to as specific questions.
* Frequent root of errors is the document search. Embedding model might not `retrieve` relevant docs. In such case we might: 1) add a keyword search, 2) train a custom embedding model for our dataset, 3) explore different similarity search methods.
* Bad responses might also be due to `insufficient docs`. In this case improving/updating docs would be useful.
* The `prompt template` can also have a significant impact on LLM responses. There are prompt engineering techniques to steer the model to generate desired outputs.
* Can experiment with different `LLM` models, or train/fine-tune a custom model.
* Can adjust sampling parameters (eg. `temperature` or `top P`)


# Strategies and ideas to enhance App

## Document search

Considered a bottleneck in many applications. 

`Problem`: Certain words (eg. proper names or new terms) might not be well represented by embedding models.
* `Solution`: Combine embedding search with keyword-based search methods.

`Problem`: Questions and answers are far away in the embedding space. Hence, difficult for embedding models to find right match.
* `Solution`: Use hypothetical document embeddings, where an LLM generates a hypothetical answer to a user question, and then the answer is embedded and used to search for relevant docs.

`Problem`: With a limited number of docs in a prompt, and large number of similar docs, the retrieved docs may lack diversity.
* `Solution`: Use maximal marginal relevance (MMR) - helps balance relevance and diversity. This increases chance that we find correct answer in the prompt.

`Problem`: Domain specific questions and docs, may not be represented well in generic models.
* `Solution`: Training a custom embedding model on your own data (designed on your domain) can improve performance


## Chains

We can experiment with different chain types available in `LangChain`. Different chain types are:

`Stuff`:
* Baseline solution.
* Stuff all related data into the prompt as context to pass to the LLM (see [2_retreival_qa_llama2.ipynb](../notebooks/2_retreival_qa_llama2.ipynb)).

`MapReduce`:
* Opportunity - Extract relevant data 
* Run an initial prompt on each chunk of data (to extract relevant info) and then `combine` the outputs in a separate prompt (to feed to LLM).

`Refine`:
* Opportunity - Refine answer with new data
* Run an initial prompt on the first chunk of data, `refine` the output based on the new document for subsequent docs.

`Map-Rerank`:
* Run an initial prompt on each chunk of data, `rank` the responses based on the certainty score and rerun the highest scored data.

Higher fidelity chain types (MapReduce, Refine, Map-Rerank) can lead to better results, but can aldo increase cost and latency.


## LLMs

There are a few providers available: OpenAI, Cohere, Anthropic, HuggingFace, Meta, Mosaic, EleutherAI.


## Prompt engineering

[Prompt engineering](https://www.promptingguide.ai/) is an important aspect of of building an LLM app (active research area). Different prompting methods include: Zero-shot, Few-shot, Chain-of-thought, Self-consistency, Generate-knowledge, Active-prompt, Directionsl Stimulus, ReAct, Multimodal CoT, Graph, Tree-of-Thoughts.


# Controlling LLM outputs

Here we cover the open source framework [Guardrails AI](https://docs.getguardrails.ai/) for controlling LLM outputs in practical applications.

## Problems 

LLMs have problems which restricts their applications when "correctness" is critical:
* Brittle and hard to control in production.
  * LLMs are stochastic (same inputs != same outputs).
  * LLMs don't follow instructions always.
* Hard to always get the correct answer (eg. hallucinations, lack of correct structure, etc.).
* Only tool available to devs is the prompt.
* LLMs are hidden behind APIs - no control over version updates.

## Solution

Combining LLMs with output verification:
* This consists of app-specific checks and verification programs, that take in the LLM output, and ensure that the LLM output is correct.
* If verification passes, we can pass output to next step.
* If verification fails, construct a new prompt with relevant context (which validation tests failed).

  ![Guardrail AI validation checks](/images/guardrailsAI_validation.png)

* Example general checks: 
  * No personally identifying info in LLM output.
  * LLM output contains no profanity.
  * LLM output does not contain names of competitor companies.
  * Check that code meets runtime limits.
  * If making summaries, make similar that summary similar to thesource

## Controlling LLMs with Guardrails AI

Guardrails AI offers:
* Framework for creating custom validators.
* Orchestration of prompting -> verification -> re-prompting.
* Library of commonly used validators for multiple use cases.
* Specification language for generating structured LLM outputs (i.e. actions to take after conducting validation tests on the LLM output). Actions include:
  * `reask`: reask LLM to generate output that meets quality criteria. The prompt used for reasking contains info about which quality criteria failed (auto-generated by the validator), and a correct answer for the failed tests is requested.
  * `fix`: programmatically fix the generated output to meet the quality criteria.
  * `filter`: filter the incorrect fields that fail. Will return the rest of the generated output.
  * `refrain`: on failure there will be a `None` output returned instead of the json.
  * `noop`: do nothing. The failure will be recorded in the logs, but no corrective action will be taken.
  * `exception`: raise an exception when validation fails.


# Safety considerations

Here we cover a self-hardening prompt injection (malicious prompt) detection framework named [Rebuff](https://github.com/protectai/rebuff). 

Example: send the LLM a query about getting (or inserting) data from a database, and the LLM generates the SQL query that the user wants. This could be used by a malicious user to retrieve user info, or insert/update new data, without the owners of the database knowing.

Self-hardening defense includes four stages:
* `Heuristic`: hardcoded rules.
* `LLM`: detect injection using LLM.
* `Semantic` use crowdsourced attack signatures.
* `Leak`: indentify attack through canary word leakage.


# Installation of libraries

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```