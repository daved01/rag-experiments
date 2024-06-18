class ConfigConstants:
    DEFAULT_CONFIG_FILE = "config.yaml"
    DEFAULT_LOGGING_FILE = "logging.yaml"
    KEY_CHUNK_SIZE = "chunk_size"
    KEY_CHUNK_OVERLAP = "chunk_overlap"
    KEY_CONFIG_DATABASE = "database"
    KEY_CONFIG_PATH = "path"
    KEY_EMBEDDING = "embedding"
    KEY_EVALUATORS = "evaluators"
    KEY_EVALUATORS_ORDER_AWARE = "order_aware"
    KEY_EVALUATORS_ORDER_UNAWARE = "order_unaware"
    KEY_EVALUATORS_ORDER_UNAWARE_K = "k"
    KEY_LOADER = "loader"
    KEY_LOCAL = "local"
    KEY_LLM = "llm"
    KEY_MAX_TOKENS = "max_tokens"
    KEY_METHOD = "method"
    KEY_OPENAI = "openai"
    KEY_PATHS = "paths"
    KEY_PIPELINES = "pipelines"
    KEY_PROMPT = "prompt"
    KEY_QUERIES = "queries"
    KEY_SPLITTER = "splitter"


class DatabaseConstants:
    KEY_DATABASE_DOCUMENTS = "documents"


class EmbeddingConstants:
    KEY_TEXT = "text"


class InputConstants:
    KEY_RELEVANT_DOCS = "relevant_docs"
    KEY_QUERIES = "queries"
    KEY_DOC = "doc"
    KEY_RELEVANCE = "relevance"


class ModelConstants:
    KEY_PAGE = "page"
    KEY_SOURCE = "source"
    KEY_TITLE = "title"
