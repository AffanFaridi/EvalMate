import logging
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

logger = logging.getLogger(__name__)

def build_llm_chain(retriever, mistral_api_key):
    logger.info("Setting up LLM and retrieval chain...")
    model = ChatMistralAI(mistral_api_key=mistral_api_key)
    prompt = ChatPromptTemplate.from_template("""
You are an expert SHL assessment recommender.

You must answer the user's question **strictly using ONLY the information in the provided context**.  
If the context does not contain information that directly answers the user's question, you MUST return:

{{
    "recommended_assessments": []
}}

**Do NOT make up, infer, or guess answers. Do NOT use your own knowledge. Do NOT return any text, explanation, or markdown-ONLY output the JSON object above.**

The required format is:

{{
    "recommended_assessments": [
        {{
            "url": "...",
            "adaptive_support": "...",
            "description": "...",
            "duration": ...,
            "remote_support": "...",
            "test_type": "...",
            "job_levels": [...],
            "languages": [...]
        }}
        // up to 10 items
    ]
}}

<context>
{context}
</context>

Question: {input}

**Instructions:**
- Only include assessments from the provided context that directly answer the user's question.
- If the user's question is not about SHL assessments, or if the context does not contain relevant information, return ONLY:
{{
    "recommended_assessments": []
}}
- NEVER add any explanation, markdown, or extra text-ONLY output the JSON object.
""")



    document_chain = create_stuff_documents_chain(model, prompt)
    logger.info("LLM and retrieval chain ready.")
    return create_retrieval_chain(retriever, document_chain)
