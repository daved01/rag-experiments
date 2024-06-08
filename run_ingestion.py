import logging

from openai_pipeline.embedding import OpenAIEmbeddings
from local_pipeline.embedding import LocalEmbeddings
from shared.database import ChromaDB
from shared.loader import Loader
from shared.models import Document
from shared.splitter import TextSplitter
from shared.utils import load_config
from shared.utils import setup_logging
from shared.constants import ConfigConstants


def main():
    print(
        "Running ingestion pipelines! Set log level to DEBUG or lower for more verbose output."
    )
    setup_logging()
    logger = logging.getLogger(__name__)

    print("Loading configuration ...")
    config = load_config(ConfigConstants.DEFAULT_CONFIG_FILE)

    loader = Loader(paths=config[ConfigConstants.KEY_LOADER][ConfigConstants.KEY_PATHS])
    documents: list[Document] = loader.load_pdf()

    splitter = TextSplitter(config=config[ConfigConstants.KEY_SPLITTER])
    docs_chunks = splitter.split_documents(documents)

    print(f"Created {len(docs_chunks)} number of document chunks!")
    method = config[ConfigConstants.KEY_SPLITTER][ConfigConstants.KEY_METHOD]
    chunk_size = config[ConfigConstants.KEY_SPLITTER][ConfigConstants.KEY_CHUNK_SIZE]
    chunk_overlap = config[ConfigConstants.KEY_SPLITTER][
        ConfigConstants.KEY_CHUNK_OVERLAP
    ]

    # OpenAI pipeline
    print("Running OpenAI pipeline ...")
    embeddings_openai = OpenAIEmbeddings(
        config[ConfigConstants.KEY_PIPELINES][ConfigConstants.KEY_OPENAI]
    )
    chunks_openai = embeddings_openai.add_embeddings_to_docs(docs_chunks)

    database_openai = ChromaDB(config, f"openai_{method}_{chunk_size}_{chunk_overlap}")
    database_openai.add_chunks(chunks_openai)

    # Local pipeline
    print("Running local pipeline ...")
    embeddings_local = LocalEmbeddings(
        config[ConfigConstants.KEY_PIPELINES][ConfigConstants.KEY_LOCAL]
    )
    chunks_local = embeddings_local.add_embeddings_to_docs(docs_chunks)

    database_local = ChromaDB(config, f"local_{method}_{chunk_size}_{chunk_overlap}")
    database_local.add_chunks(chunks_local)

    print("Done!")


if __name__ == "__main__":
    main()
