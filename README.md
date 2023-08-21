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

# Notebooks

The notebooks in this repository are related to the topics covered in the [Theory of LLMs section](#theory-of-llms):
* [0_using_APIs.ipynb](/notebooks/0_using_APIs.ipynb): covers [tokenisation](#predicting-the-next-token), calling an LLM from an API, and [sampling methods](#sampling-methods).
* [1_generation.ipynb](/notebooks/1_generation.ipynb): covers [prompt engineering](#prompting-structure) (zero shot, few shot, adding context & response, and Level 5 prompting) for building a chatbot.
* [2_retreival_qa.ipynb](/notebooks/2_retreival_qa.ipynb): covers 
* 
# Theory of LLMs

For details about the transformer architecture of LLMs, click the [link](https://jalammar.github.io/illustrated-gpt2/).

## Predicting the next token. 

Basic process:
1) Start with some input text.
2) Split the text into tokens, represented by numbers.
3) Feed chain of tokens (current and past ones) into LLM.
4) LLM returns distribution of probabilities over the entire vocabulary (i.e. all the available tokens to our model), for the next token to continue the sequence.
5) We can select (*sample*) the token with the highest probability for example (i.e. greedy decoding).
6) Append selected token to the input sequence and we repeat the process (i.e. tokanize input sequence, feed to LLM, output vocab probabilities and select next the token).


## Sampling methods

LLMs generate text through sampling. They produce a set of probabilities across the vocabulary, and pick a toke to follow the input sequence. Different methods of sampling are:
* **Greedy decoding** 
 * Pick tocken with highest probability.
* **Beam search** 
 * Generate multiple candidate sequences to maximise probability of a sequence of tokens. Often leads to repetitions and lack of meaning.
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
   * In this phase, the model can be good at predicting text text found in the pre-train dataset.
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
* Chunk docs using a `sliding window`. 
* `Overlapping` chunks to ensure we don't loose useful information (eg. cutting in half sentences or code snippets). 
* Use semantic structure of docs (eg. markdown headers to guide division of docs into chunks).
* Might need to deal with different document types (eg. domain-specific text, unique formats like latex, code, pdfs).
  * `Longchain` library might be useful - since contains various utility functions to handle different document types.


<!-- ## Assembling components

Check this [link](https://github.com/wandb/wandbot) for wandbot integration into Slack and Discord integrations. -->

