import logging
from src.data_converter import convert_json_to_docs

logger = logging.getLogger(__name__)

def load_docs():
    logger.info("Loading documents...")
    docs = convert_json_to_docs("data/shl_recommended_assessments.json")
    logger.info(f"Loaded {len(docs)} documents.")
    return docs
