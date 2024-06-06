import json
import time
import logging.config
import yaml


def load_prompt_queries(query_file):
    with open(query_file, "r") as file:
        return json.load(file)


def load_config(config_file: str) -> dict:
    """Loads the config."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def setup_logging(config_file="logging.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)


def save_results(config: dict, results: dict) -> None:
    """Saves results to a file."""
    path = config.get("directory", "results")
    output_file = f"{path}/results_{int(time.time())}.json"
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)


def create_prompt(template, query: str, contexts: list[str]) -> str:
    """Constructs a prompt from a template, a query, and a list of contexts."""
    contexts_strs = "|".join(contexts)
    prompt = template.format(query=query, contexts=contexts_strs)
    print(prompt)
    return prompt
