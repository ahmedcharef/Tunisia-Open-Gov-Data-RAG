#!/usr/bin/env python3
"""
Tunisia Education RAG - Streamlit Web Interface

Features:
- Multi-tab interface (Chat + Statistics Dashboard)
- Dynamic governorate filtering
- Source citations
- Shared RAGService layer (same logic as CLI)
- Robust error handling
"""

import streamlit as st
from dotenv import load_dotenv
from typing import List

from src.config import Config
from src.rag_service import RAGService
from src.retriever import get_governorate_breakdown, get_vectorstore_stats, get_available_governorates

load_dotenv()

st.set_page_config(
    page_title="🇹🇳 Tunisia Multi-Dataset RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 Tunisia Multi-Dataset RAG")
st.markdown("**Intelligent Assistant for Tunisian Open Government Data**")
st.caption("Data from data.gov.tn")

# ====================== SESSION STATE ======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[dict] = []

if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = Config.DEFAULT_DATASET

# Initialize RAG service
@st.cache_resource
def get_rag_service(dataset: str):
    return RAGService(dataset=dataset)

service = get_rag_service(st.session_state.current_dataset)

# Load available governorates for current dataset
available_govs = ["All"] + get_available_governorates(st.session_state.current_dataset)

# ====================== TABS ======================
tab_chat, tab_stats = st.tabs(["💬 Chat Assistant", "📊 Statistics Dashboard"])

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Dataset Selector
    selected_dataset = st.selectbox(
        "Select Dataset",
        options=list(Config.DATASETS.keys()),
        index=list(Config.DATASETS.keys()).index(st.session_state.current_dataset),
        key="dataset_selector"
    )

    # Update service if dataset changed
    if selected_dataset != st.session_state.current_dataset:
        st.session_state.current_dataset = selected_dataset
        st.rerun()

    k_value = st.slider("Number of documents to retrieve (k)", min_value=4, max_value=20, value=8)

    st.markdown("### Governorate Filter")
    selected_gov = st.selectbox(
        "Governorate",
        options=available_govs,
        index=0,
        key="gov_selector"
    )

    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Model: **{Config.OPENROUTER_MODEL if Config.LLM_PROVIDER == 'openrouter' else Config.OLLAMA_MODEL}**")

# ====================== TAB 1: CHAT ASSISTANT ======================
with tab_chat:
    st.subheader(f"Chat - {st.session_state.current_dataset.title()} Dataset")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about educational institutions in Tunisia..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the database..."):
                try:
                    # Pass selected governorate to the service
                    gov_filter = None if selected_gov == "All" else selected_gov

                    result = service.query(
                        user_input=prompt,
                        chat_history=st.session_state.chat_history[:-1],
                        k=k_value,
                        gouvernorat=gov_filter
                    )

                    st.markdown(result["answer"])

                    if result.get("sources"):
                        st.markdown("**📚 Sources:**")
                        for source in result["sources"]:
                            st.caption(f"• {source}")

                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": result["answer"]
                    })

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# ====================== TAB 2: STATISTICS DASHBOARD ======================
with tab_stats:
    st.subheader(f"📊 Statistics - {st.session_state.current_dataset.title()} Dataset")

    try:
        stats = get_vectorstore_stats(st.session_state.current_dataset)
        total_docs = stats.get('total_documents', 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Establishments", f"{total_docs:,}")
        with col2:
            st.metric("Collection", Config.get_collection_name(st.session_state.current_dataset))
        with col3:
            st.metric("Status", "✅ Ready")

        st.markdown("---")

        st.subheader("Governorate Distribution")
        df_gov = get_governorate_breakdown(st.session_state.current_dataset)

        if not df_gov.empty and len(df_gov) > 0:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.bar_chart(df_gov.set_index("Governorate")[:15])
            with col2:
                st.dataframe(
                    df_gov.head(15),
                    width='stretch',
                    hide_index=True
                )
            st.caption(f"Showing top {min(15, len(df_gov))} governorates")
        else:
            st.info("No governorate breakdown data available.")

        st.markdown("---")

        st.subheader("Key Insights")
        st.info(f"""
        • **Total records indexed**: {total_docs:,}  
        • Data includes public universities, public schools, and private schools  
        • Governorate filtering is dynamic based on actual loaded data  
        • Source citations are shown for transparency
        """)

    except Exception as e:
        st.error(f"Failed to load statistics: {e}")

# Footer
st.markdown("---")
st.caption(
    "🇹🇳 Tunisia Open Government Data RAG | "
    "Multi-Dataset Support | Built with LangChain + Streamlit"
)