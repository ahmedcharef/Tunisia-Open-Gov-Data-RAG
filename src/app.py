import streamlit as st
from dotenv import load_dotenv
from typing import List

from src.config import Config
from src.rag_service import RAGService
from src.retriever import get_vectorstore_stats

load_dotenv()

st.set_page_config(
    page_title="🇹🇳 Tunisia Education RAG",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎓 Tunisia Education RAG")
st.markdown("**Intelligent Assistant for Tunisian Educational Institutions**")
st.caption("Official data from data.gov.tn")

# ====================== SESSION STATE ======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[dict] = []

# Initialize shared RAG service
@st.cache_resource
def get_rag_service():
    return RAGService()

service = get_rag_service()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Settings")
    
    k_value = st.slider(
        "Number of documents to retrieve (k)", 
        min_value=4, 
        max_value=20, 
        value=8,
        help="Higher values provide more context but may slow down responses"
    )

    st.markdown("### Governorate Filter")
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

# ====================== CHAT INTERFACE ======================
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about educational institutions in Tunisia..."):
    
    # Add user message to history and display
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the database..."):
            try:
                # Use shared service
                result = service.query(
                    user_input=prompt, 
                    chat_history=[(m["role"], m["content"]) for m in st.session_state.chat_history[:-1]],
                    k=k_value
                )

                # Display answer
                st.markdown(result["answer"])

                # Display sources if available
                if result.get("sources"):
                    st.markdown("**📚 Sources:**")
                    for source in result["sources"]:
                        st.caption(f"• {source}")

                # Add assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": result["answer"]
                })

            except Exception as e:
                st.error(f"An error occurred while generating the response: {str(e)}")

# Footer
st.markdown("---")
st.caption(
    "🇹🇳 Tunisia Open Government Data RAG | "
    "Data source: data.gov.tn | "
    "Built with LangChain + Streamlit"
)