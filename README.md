# RAG Experiments

<p align="center">
  <a href="https://www.python.org/downloads/release/python-310/"><img src="https://img.shields.io/badge/python-3.10-green.svg" alt="Python 3.10"></a>
  <a href="https://www.python.org/downloads/release/python-311/"><img src="https://img.shields.io/badge/python-3.11-green.svg" alt="Python 3.11"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

A repository for running experiments on Retrieval-Augmented Generation (RAG) pipelines.

This codebase is designed for conducting experiments and tutorials on improving RAG pipelines. It is built for flexibility and modularity, allowing easy swapping and observation of components to measure performance. This is not intended to be a production RAG system. For production-ready solutions, please contact me.

I am working on a blog post series and associated experiments. Links will be added here once they are published. More information can be found on my website [DeconvoluteAI](https://deconvoluteai.com).

## Quick Start

Before you start, make sure to have your OpenAI API key set under `OPENAI_API_KEY`, if you want to run the OpenAI pipeline.

1. Install dependencies:

```bash
pip install -r requirements.txt
```

If you also want to use a local LLAMA3 model, you have to install OLLAMA and download the model with `ollama pull llama3`. To do so, follow [these instructions](https://github.com/ollama/ollama). If you don't want to run the local pipeline, just comment it out.

2. Configure the app settings in the config.yaml file.

3. Ingest data into your database:

```bash
python run_ingestion.py
```

A database is created in `data/db`.

4. Add queries to the prompts_queries.json file and run experiments:

```bash
python run_experiments.py
```

Results will be saved as `JSON` files in the `data/results` directory.

## Installation

Ensure you have all the required dependencies installed by running:

```bash
pip install -r requirements.txt
```

You might have to manually create two directories: `data` and `data/results`.

As mentioned above, the local pipeline uses OLLAMA through Langchain to access `LLAMA3 8B`. Make sure you have OLLAMA installed and the llama model downloaded.

## Running Experiments

Once installed you can run experiments on the pipelines. To configure them, use the configuration files.

### Configuration Files

- `config.yaml`: Contains settings and parameters for running experiments.
- `prompt_queries.json`: Holds the prompt template and a list of queries for the experiments, as well as the ground-truth documents for evaluation. Each query triggers a separate run of all pipelines.

### Adding Evaluators

Evaluators are defined in `evaluators`. The top-level class `RetrievalEvaluator` is an evaluator service, to which all evaluators you want to run are added in the `run()` method.

Each evaluator must implement the `BaseEvaluator` and makes use of a number of metrics. All evaluator `run()` methods accept an `ExperimentResults` object with or without existing evaluations. The methods return the same object with evaluations appended.

Once all evaluators are defined you can add your evaluator service in `run_experiment()` like this:

```Python
# Initialize with config and experiment inputs.
retrieval_evaluators = RetrievalEvaluator(config, prompts_queries)

# Run evaluators
results_with_evals = retrieval_evaluators.run(results_with_or_without_evals)
```

### Running Scripts

- `run_ingestion.py`: Script for ingesting data into the database.
- `run_experiments.py`: Script for running experiments based on the queries defined in prompt_queries.json.

## Development

The project is modular and meant to be extended. You can add advanced methods to the existing pipeline or introduce new pipelines. The structure allows for easy modifications and enhancements.

### File Structure

- `run_*.py`: Main code that brings together the other modules to run ingestion or experiments.
- `shared`: Contains code that is shared of each component, including loader and splitter. Note that the database uses collections to separate data for each pipeline.
- `openai_pipeline`: Code specific to the OpenAI pipeline.
- `local_pipeline`: Code specific to the local pipeline.
- `evaluators`: Code for evaluators.

Each pipeline contains two parts.

- `llm.py`: Handles LLM interactions, including the prompt.
- `embedding.py`: Manages embeddings.

The pipeline is then defined in the `__init__.py`, for example `OpenAIPipeline` using the components.

### OpenAI pipeline

This pipeline uses the OpenAI embedding model `text-embedding-3-small`. It has an input token [limit](https://platform.openai.com/docs/guides/embeddings/use-cases) of `8191`, and returns a vector of length `1536`.

The number of tokens of an input is checked using `tiktoken`.

### Local pipeline

The local pipeline uses the sentence transformer embedding model `sentence-transformers/all-MiniLM-L6-v2`, see [here](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It has an input token [limit](TODO) of `512` tokens, and returns a `384` dimensional array.

To check the number of input tokens, the `Tokenizer` uses the `AutoTokenizer.from_pretrained(<model_name>)` method.

## Contribute

At this time, the project is primarily intended for running experiments. Contributions for bug fixes and minor improvements are welcome. For substantial changes or feature additions, please contact me first to discuss the proposed changes.

## Contact

For more information or inquiries about production systems, please contact me through my website [DeconvoluteAI](https://deconvoluteai.com/contact).
