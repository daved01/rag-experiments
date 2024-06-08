import json
import logging.config
import yaml
from dataclasses import asdict
from datetime import datetime

from shared.constants import ConfigConstants
from shared.models import ExperimentResults


def load_prompt_queries(query_file):
    with open(query_file, "r") as file:
        return json.load(file)


def load_config(config_file: str) -> dict:
    """Loads the config."""
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def setup_logging(config_file=ConfigConstants.DEFAULT_LOGGING_FILE):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
        logging.config.dictConfig(config)


def create_prompt(template, query: str, contexts: list[str]) -> str:
    """Constructs a prompt from a template, a query, and a list of contexts."""
    contexts_strs = "|".join(contexts)
    prompt = template.format(query=query, contexts=contexts_strs)
    return prompt


def experiment_results_to_dict(experiment_results: ExperimentResults) -> dict:
    """Converts an `ExperimentResults` object to a dict."""

    def serialize(obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, list):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {key: serialize(value) for key, value in obj.items()}
        return obj

    return serialize(asdict(experiment_results))


def save_experiments_results_to_json(
    experiment_results: list[ExperimentResults], base_path: str
):
    """Saves a list of `ExperimentResults` objects to a json file."""
    data_list = [experiment_results_to_dict(er) for er in experiment_results]

    curr_time = datetime.now()
    formatted_time = curr_time.strftime("%Y-%m-%d_%H-%M-%S")
    file_path = f"{base_path}/results_{formatted_time}.json"
    with open(file_path, "w") as json_file:
        json.dump(data_list, json_file, indent=4)
    print(f"Saved results to file: {file_path}!")
