import logging

logger = logging.getLogger(__name__)

def build_retriever(vectorstore, k=25):
    logger.info(f"Building retriever with top-k={k}")
    return vectorstore.as_retriever(search_kwargs={"k": k})
