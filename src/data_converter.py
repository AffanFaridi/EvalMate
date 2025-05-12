import json
import logging
from langchain.schema import Document

logger = logging.getLogger(__name__)

def convert_json_to_docs(json_path):
    logger.info(f"Loading JSON data from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assessments = data.get("recommended_assessments", data)
    docs = []
    for idx, item in enumerate(assessments):
        metadata = {}
        for key, value in item.items():
            value_str = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            metadata[key] = value_str
        lines = []
        for key, value in item.items():
            if key == "url":
                continue
            value_str = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
            lines.append(f"{key}: {value_str}")
        page_content = " | ".join(lines)
        docs.append(Document(page_content=page_content, metadata=metadata))
        logger.debug(f"Processed doc {idx+1}: {metadata.get('url', '')}")
    logger.info(f"Converted {len(docs)} documents.")
    return docs
