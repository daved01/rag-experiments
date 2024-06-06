# RAG Experiments

<p align="center">
  <a href="https://www.python.org/downloads/release/python-310/"><img src="https://img.shields.io/badge/python-3.10-green.svg" alt="Python 3.10"></a>
  <a href="https://www.python.org/downloads/release/python-311/"><img src="https://img.shields.io/badge/python-3.11-green.svg" alt="Python 3.11"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

A repo for running experiments on Retrieval-Augmented Generation pipelines.

This code is meant to be used for experiments and tutorials about how to improve RAG pipelines. That is why it is built the way it is built. It is not async, because it does not have to be. It is modular to easily swap out components or to tap into them to observe performance. This is not an attempt to build a production RAG system. If you are looking for that, contact me.

I am working on the blog post series and the experiments for it and will be adding links here once they are published. More infos on my website [DeconvoluteAI](https://deconvoluteai.com)

# Quick Start

Run `pip install -r requirements`.

Configure the app in the `config.yaml` file.

Add queries to run in the `queries.json` file. Queries are run sequentially.

Then you can run the ingestion script `python run_ingestion.py` to ingest data into your database.

Once you have data, run `python run_experiments.py`.

You will get a file with the results.
