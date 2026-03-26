#!/usr/bin/env python3
"""
Tunisia Education RAG - Streamlit Web Interface

This is the main web application for the Tunisia Education RAG project.
It provides two tabs:
1. Chat Assistant - Natural language querying over Tunisian educational institutions
2. Statistics Dashboard - Overview and metrics of the loaded database

Features:
- Governorate-based filtering
- Source citations with metadata
- Shared RAG service layer (same logic as CLI)
- Clean multi-tab interface
"""

import streamlit as st
from dotenv import load_dotenv
from typing import List
import pandas as pd

from src.config import Config
from src.rag_service import RAGService
from src.retriever import get_vectorstore_stats, get_available_governorates, get_governorate_breakdown

load_dotenv()

st.set_page_config(
    page_title="🇹🇳 Tunisia Education RAG",
    page_icon="🎓",
    layout="wide",
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

# Load available governorates for filter
available_govs = ["All"] + get_available_governorates()

# ====================== TABS ======================
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 Statistics Dashboard"])

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
        options=available_govs,
        index=0
    )

    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption(
        f"Model: **{Config.OPENROUTER_MODEL if Config.LLM_PROVIDER == 'openrouter' else Config.OLLAMA_MODEL}**"
    )

# ====================== TAB 1: CHAT ASSISTANT ======================
with tab1:
    st.subheader("Chat with the Assistant")

    # Display chat history
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
                    result = service.query(
                        user_input=prompt,
                        chat_history=st.session_state.chat_history[:-1],
                        k=k_value
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
with tab2:
    st.subheader("📊 Database Statistics")

    try:
        stats = get_vectorstore_stats()
        total_docs = stats.get('total_documents', 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Establishments", f"{total_docs:,}")
        with col2:
            st.metric("Collection", Config.COLLECTION_NAME)
        with col3:
            st.metric("Status", "✅ Ready")

        st.markdown("---")

        # Governorate Breakdown
        st.subheader("Governorate Distribution")
        df_gov = get_governorate_breakdown()

        if not df_gov.empty:
            col1, col2 = st.columns([3, 2])
            with col1:
                st.bar_chart(df_gov.set_index("Governorate")[:15])
            with col2:
                st.dataframe(
                    df_gov.head(15),
                    use_container_width=True,
                    hide_index=True
                )
            
            st.caption(f"Showing top governorates from a sample of up to 3,000 records.")
        else:
            st.info("No governorate information could be extracted from the data.")

        st.markdown("---")

        st.subheader("Key Insights")
        st.info(f"""
        - **Total records indexed**: {total_docs:,}
        - Data includes public universities, public schools, and private schools
        - Governorate filtering is dynamic based on actual loaded data
        """)

    except Exception as e:
        st.error(f"Failed to load statistics: {e}")

# Footer
st.markdown("---")
st.caption(
    "🇹🇳 Tunisia Open Government Data RAG | "
    "Data source: data.gov.tn | "
    "Built with LangChain + Streamlit"
)
