# RAG Baseline with OpenAI

Code for a baseline RAG pipeline with the OpenAI API. You can find the post [here](TODO: Link post).

# Installation

Make sure you have at least Python 3.10 installed.

Before you can run the code, install the dependecies with `pip install -r requirements.txt`.

# Getting Started

Before you run it, make sure you have you OpenAI API key set to `OPENAI_API_KEY` as explained [here](TODO: Link openai).

Then, run the program with `python run.py` in the root.

## Structure

The following structure exists.

```bash
openai/
    ├── embedding.py        # Handles embedding generation
    ├── retriever.py        # Base retriever class and methods
    └── llm.py              # Interface with the language model
├── config.yaml             # Configuration file for hyperparameters and settings
├── queries.json            # Queries to be run with their IDs
├── database.py             # Manages data storage and retrieval
├── loader.py               # Handles data loading and preprocessing
├── run_ingestion.py        # Script to ingest data
└── run_experiments.py      # Script to run experiments
```

## File Descriptions

`run.py`: Main entry point. Orchestrates the ingestion, retrieval, and synthesis phases by calling functions from the other modules.

`llm.py`: Handles interactions with the language model, including defining prompts and making API calls.

`embedding.py`: Generates and manages embeddings using the OpenAI API.

`retriever.py`: Handles interactions with the database (Chroma in this case) to store and retrieve embeddings/document chunks.

# Architecture

Database
One database for both. Use collections to separate.
