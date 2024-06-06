import logging

from openai_models.embedding import OpenAIEmbeddings
from shared.database import ChromaDB
from shared.loader import Loader
from shared.models import Document
from shared.splitter import TextSplitter
from shared.utils import load_config
from shared.utils import setup_logging


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Loading configuration ...")
    config = load_config("config.yaml")

    loader = Loader(paths=config["loader"]["paths"])
    documents: list[Document] = loader.load_pdf()

    splitter = TextSplitter(config=config["splitter"])
    docs_chunks = splitter.split_documents(documents)

    # OpenAI
    embeddings_openai = OpenAIEmbeddings(config["pipelines"]["openai"])
    chunks_openai = embeddings_openai.add_embeddings_to_docs(docs_chunks)

    method = config["splitter"]["method"]
    chunk_size = config["splitter"]["chunk_size"]
    chunk_overlap = config["splitter"]["chunk_overlap"]
    database_openai = ChromaDB(config, f"openai_{method}_{chunk_size}_{chunk_overlap}")
    database_openai.add_chunks(chunks_openai)

    # TODO: Local
    # embedding_local = None
    # chunks_local = embedding_local.embed(docs_chunks)
    # database_local = ChromaDB(config, f"local_{config["splitter"]["method"]}_{config["splitter"]["chunk_size"]}_{config["splitter"]["chunk_overlap"]}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
