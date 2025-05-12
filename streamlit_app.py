import streamlit as st
from dotenv import load_dotenv
import os
import json
import logging
import re
from src.data_loader import load_docs
from src.vector_store_setup import load_vectorstore
from src.retriever_setup import build_retriever
from src.llm_chain import build_llm_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def extract_json(text):
    match = re.search(r'({[\s\S]*})', text)
    if match:
        return json.loads(match.group(1))
    raise ValueError("No JSON found in response")

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource(show_spinner="Initializing AI engine...")
def load_resources():
    try:
        logger.info("Loading documents...")
        docs = load_docs()
        logger.info("Connecting to Pinecone...")
        vectorstore = load_vectorstore("shl2")
        logger.info("Building retriever...")
        retriever = build_retriever(vectorstore, k=10)
        logger.info("Initializing LLM chain...")
        chain = build_llm_chain(retriever, os.getenv("MISTRAL_API_KEY"))
        logger.info("Resource loading complete")
        return chain
    except Exception as e:
        logger.error(f"Resource loading failed: {str(e)}")
        st.error(f"Initialization failed: {str(e)}")
        st.stop()

def main():
    st.image("https://www.shl.com/wp-content/themes/shl/images/logo.png", width=120)
    st.title("SHL Assessment Recommendation Engine 🚀")
    st.markdown("Find the perfect SHL assessments for your needs. Enter your requirements below.")

    with st.sidebar:
        st.header("ℹ️ About")
        st.info("This app recommends SHL assessments based on your needs. Powered by LangChain, Pinecone, and MistralAI.")

    try:
        with st.spinner("Loading AI engine (first time may take a minute)..."):
            chain = load_resources()
            if not os.getenv("MISTRAL_API_KEY"):
                raise ValueError("Mistral API key not found")
            if not os.getenv("PINECONE_API_KEY"):
                raise ValueError("Pinecone API key not found")
            st.success("System ready! Enter your query below")

        with st.form("query_form"):
            query = st.text_input(
                "Enter your assessment requirements:",
                placeholder="e.g., 'Find programming tests for mid-level candidates in French'",
                key="query_input"
            )
            submit = st.form_submit_button("🔎 Search")

        if submit and query:
            with st.spinner("Searching 500+ SHL assessments..."):
                try:
                    result = chain.invoke({"input": query})
                    raw_response = result["answer"].strip()
                    if not raw_response:
                        raise ValueError("Empty LLM response")
                    logger.debug(f"Raw response: {raw_response}")
                    try:
                        response = extract_json(raw_response)
                    except json.JSONDecodeError:
                        cleaned = raw_response.replace("'", '"').replace("True", "true").replace("False", "false")
                        response = extract_json(cleaned)
                    if "recommended_assessments" not in response:
                        raise KeyError("Missing required field: recommended_assessments")
                    if not response["recommended_assessments"]:
                        st.warning("No matching assessments found")
                        return
                    st.success(f"Found {len(response['recommended_assessments'])} recommendations")
                    for idx, assessment in enumerate(response["recommended_assessments"], 1):
                        with st.expander(f"Assessment #{idx}: {assessment['description'][:80]}..."):
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown(f"""
                                **Description**: {assessment["description"]}  
                                **Duration**: {assessment["duration"]} mins  
                                **Remote Support**: {assessment["remote_support"]}  
                                **Test Type**: {assessment["test_type"]}
                                """)
                            with cols[1]:
                                st.markdown(f"""
                                **Job Levels**: {", ".join(assessment["job_levels"])}  
                                **Languages**: {", ".join(assessment["languages"])}
                                """)
                            if assessment.get("url"):
                                st.markdown(f"[View Assessment]({assessment['url']})")
                    logger.info("Results displayed successfully")
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    st.error(f"Error processing response: {str(e)}")
                    st.code(raw_response, language='text')
    except Exception as e:
        logger.critical(f"Critical failure: {str(e)}")
        st.error(f"System error: {str(e)}")

if __name__ == "__main__":
    main()
