
# SHL Assessment Recommender (EvalMate)

**EvalMate** is an intelligent assistant that helps you discover the best SHL assessments for your hiring and evaluation needs-instantly, accurately, and interactively.

With EvalMate, you can simply describe your requirements in **natural language**-for example:
- "Show me programming assessments for intermediate candidates in French."
- "Find the fastest English cognitive tests for sales roles."
- "I need a 30-minute test for entry-level finance positions."

EvalMate understands your intent and context, and recommends only those SHL assessments that truly match your needs, including:
- **Knowledge area** (e.g., programming, sales, finance, cognitive ability)
- **Assessment duration** (e.g., "under 30 minutes", "short tests")
- **Language** (e.g., English, French, Spanish, etc.)
- **Seniority level** (e.g., entry, mid, senior)
- **Other custom criteria** you specify in plain English

Behind the scenes, EvalMate uses advanced retrieval-augmented generation (RAG) with Pinecone vector search, HuggingFace sentence embeddings, and MistralAI LLMs-ensuring recommendations are **relevant, grounded, and never hallucinated**.

---

## 🚀 Features

- **Streamlit web UI** for user-friendly queries and result browsing
- **Retrieval-augmented generation**: combines vector search (Pinecone) with LLM reasoning (MistralAI)
- **Context-grounded answers**: Only recommends assessments present in your SHL JSON data
- **No hallucinations**: LLM is strictly instructed to answer only from provided context
- **Highly configurable**: Easily swap vector DB, embedding model, or LLM

---

## 🗂️ Project Structure

```
.
├── data/
│   └── shl_recommended_assessments.json
├── src/
│   ├── data_converter.py
│   ├── data_loader.py
│   ├── llm_chain.py
│   ├── retriever_setup.py
│   └── vector_store_setup.py
├── streamlit_app.py
├── requirements.txt
├── setup.py
└── .env (you create this)
```

---

## ⚙️ Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/AffanFaridi/EvalMate.git
   cd EvalMate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your environment variables:**

   Create a `.env` file in the project root with your API keys:
   ```
   MISTRAL_API_KEY=your-mistral-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   ```

---

## 🚦 Usage

### **To launch the Streamlit app:**
```bash
streamlit run streamlit_app.py
```

- The app will load your SHL assessment data, connect to Pinecone, and initialize the LLM.
- Enter your requirements (e.g., "Find programming tests for mid-level candidates in French") and get tailored recommendations.

---

## 📝 How It Works

1. **Data Loading:**  
   - `data_loader.py` loads and converts the SHL JSON data into LangChain `Document` objects via `data_converter.py`.

2. **Vector Store Setup:**  
   - `vector_store_setup.py` builds or loads a Pinecone vector store using HuggingFace sentence embeddings.

3. **Retriever Setup:**  
   - `retriever_setup.py` creates a retriever for semantic search.

4. **LLM Chain:**  
   - `llm_chain.py` sets up a LangChain chain with MistralAI, using a strict prompt to only answer from context.

5. **Streamlit UI:**  
   - `streamlit_app.py` ties everything together, providing a web interface for user queries and displaying recommendations.

---

## 🧩 Dependencies

- langchain, langchain-core, langchain-pinecone, langchain-huggingface, langchain-mistralai
- pinecone-client
- sentence-transformers
- streamlit
- python-dotenv
- orjson (optional, for robust JSON handling)

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Customization

- **To add or update SHL assessments:**  
  Edit `data/shl_recommended_assessments.json` and rerun the app.
- **To change the number of retrieved documents:**  
  Edit the `k` parameter in `retriever_setup.py` and `llm_chain.py`.

---

## 🤝 Contributing

Pull requests and issues are welcome!  
If you add new features or fix bugs, please open a PR.

---

## 📄 License

MIT License

---

## 🙋 Contact

Created by [Affan Faridi](https://github.com/AffanFaridi)  
For issues, use GitHub Issues or reach out via your profile.
