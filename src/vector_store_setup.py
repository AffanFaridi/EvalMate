import logging
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def build_vectorstore(docs, index_name="shl2"):
    logger.info("Creating embeddings and upserting to Pinecone...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = PineconeVectorStore.from_documents(
        docs,
        embedding=embedding,
        index_name=index_name
    )
    logger.info(f"Vectorstore built and {len(docs)} docs upserted to index '{index_name}'.")
    return vectorstore

def load_vectorstore(index_name="shl2"):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    logger.info(f"Loading existing Pinecone vectorstore index '{index_name}'...")
    return PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embedding)
