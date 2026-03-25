import streamlit as st
from dotenv import load_dotenv
from typing import List

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from src.config import Config
from src.prompts import get_contextualize_prompt, get_qa_prompt
from src.retriever import get_retriever
from src.utils import extract_gouvernorat

load_dotenv()

st.set_page_config(
    page_title="🇹🇳 Tunisia Education RAG",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎓 Tunisia Education RAG")
st.markdown("**Intelligent Assistant for Tunisian Schools & Universities**")
st.caption("Official data from data.gov.tn")

# ====================== SESSION STATE ======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[dict] = []

if "vectorstore" not in st.session_state:
    try:
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        st.session_state.vectorstore = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=Config.COLLECTION_NAME,
        )
        st.success("✅ Database loaded successfully", icon="✅")
    except Exception as e:
        st.error(f"❌ Failed to load database: {e}")
        st.stop()

# ====================== LLM SETUP (Fixed warning) ======================
@st.cache_resource
def get_llm():
    try:
        if Config.LLM_PROVIDER == "openrouter":
            return ChatOpenAI(
                model=Config.OPENROUTER_MODEL,
                api_key=Config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.25,
                max_tokens=2048,
                model_kwargs={
                    "extra_headers": {
                        "HTTP-Referer": "https://github.com/ahmedcharef/Tunisia-Open-Gov-Data-RAG",
                        "X-Title": "Tunisia Education RAG",
                    }
                },
            )
        else:
            return ChatOllama(
                model=Config.OLLAMA_MODEL,
                temperature=0.25,
                num_ctx=32768,
            )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        st.stop()


llm = get_llm()

# ====================== PROMPTS ======================
contextualize_prompt = get_contextualize_prompt()
qa_prompt = get_qa_prompt()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Settings")
    
    k_value = st.slider("Number of documents to retrieve (k)", min_value=4, max_value=20, value=8)
    
    st.markdown("### Filter by Governorate")
    selected_gov = st.selectbox(
        "Governorate",
        options=["All", "TUNIS", "SFAX", "SOUSSE", "ARIANA", "BEN AROUS", "MANOUBA",
                 "NABEUL", "BIZERTE", "MONASTIR", "MAHDIA", "KAIROUAN", "GAFSA", "MEDENINE"],
        index=0
    )

    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Model: **{Config.OPENROUTER_MODEL if Config.LLM_PROVIDER == 'openrouter' else Config.OLLAMA_MODEL}**")

# ====================== MAIN CHAT INTERFACE ======================
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Tunisian educational institutions..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the database..."):
            try:
                current_retriever = get_retriever(
                    k=k_value, 
                    gouvernorat=selected_gov if selected_gov != "All" else None
                )

                history_aware = create_history_aware_retriever(llm, current_retriever, contextualize_prompt)
                chain = create_retrieval_chain(history_aware, create_stuff_documents_chain(llm, qa_prompt))

                response = chain.invoke({
                    "input": prompt,
                    "chat_history": [(m["role"], m["content"]) for m in st.session_state.chat_history[:-1]]
                })

                answer = response["answer"]
                st.markdown(answer)

                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"An error occurred while generating the response: {str(e)}")

st.markdown("---")
st.caption("🇹🇳 Tunisia Open Government Data RAG | "
"Built with LangChain + Streamlit"
)