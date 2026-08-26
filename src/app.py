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

# ====================== CACHED HELPERS ======================
@st.cache_resource
def get_rag_service(dataset: str):
    return RAGService(dataset=dataset)

@st.cache_data(ttl=300, show_spinner=False)
def cached_governorates(dataset: str) -> List[str]:
    return get_available_governorates(dataset)

@st.cache_data(ttl=300, show_spinner=False)
def cached_stats(dataset: str):
    return get_vectorstore_stats(dataset)

@st.cache_data(ttl=300, show_spinner=False)
def cached_breakdown(dataset: str, breakdown_col: str):
    return get_governorate_breakdown(dataset, breakdown_col=breakdown_col)

service = get_rag_service(st.session_state.current_dataset)

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
        options=["All"] + cached_governorates(st.session_state.current_dataset),
        index=0,
        key="gov_selector"
    )

    if st.button("🔄 Reset Conversation"):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("🔄 Refresh Stats"):
        cached_stats.clear()
        cached_breakdown.clear()
        cached_governorates.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"Model: **{Config.OPENROUTER_MODEL if Config.LLM_PROVIDER == 'openrouter' else Config.OLLAMA_MODEL}**")

# ====================== TAB 1: CHAT ASSISTANT ======================
with tab_chat:
    st.subheader(f"Chat - {st.session_state.current_dataset.title()} Dataset")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask a question about {st.session_state.current_dataset} data in Tunisia..."):
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
    dataset = st.session_state.current_dataset
    ui_cfg = Config.DATASET_UI.get(dataset, {})
    primary_metric = ui_cfg.get("primary_metric", "Indexed Chunks")
    breakdowns = ui_cfg.get("breakdowns", [])

    st.subheader(f"📊 Statistics — {dataset.title()} Dataset")

    try:
        stats = cached_stats(dataset)
        total_docs = stats.get('total_documents', 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Indexed Chunks", f"{total_docs:,}")
        with col2:
            row_count = stats.get("source_row_count")
            st.metric("Source Rows", f"{row_count:,}" if row_count is not None else "—")
        with col3:
            status = stats.get("status", "unknown")
            st.metric("Status", "✅ Ready" if status == "ready" else f"⚠️ {status}")
        st.caption("ℹ️ Source Rows = original records from your data files. Indexed Chunks = text segments stored in the vector database after splitting (one row may produce multiple chunks).")

        st.markdown("---")

        # ── Breakdown charts — one per entry in Config.DATASET_UI[dataset]['breakdowns'] ──
        if breakdowns:
            for bd in breakdowns:
                col_key   = bd.get("col")
                col_label = bd.get("label", col_key)

                st.subheader(f"{col_label} Distribution")
                df_breakdown = cached_breakdown(dataset, col_key)

                if not df_breakdown.empty:
                    first_col = df_breakdown.columns[0]
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.bar_chart(df_breakdown.set_index(first_col)[:15])
                    with col2:
                        st.dataframe(
                            df_breakdown.head(15),
                            use_container_width=True,
                            hide_index=True,
                        )
                    st.caption(
                        f"Showing top {min(15, len(df_breakdown))} out of "
                        f"{len(df_breakdown)} {col_label.lower()} values"
                    )
                else:
                    st.info(
                        f"No {col_label.lower()} data available. "
                        "Make sure ingestion has been run for this dataset."
                    )
        else:
            st.info(f"No breakdown charts configured for the **{dataset}** dataset.")

        st.markdown("---")

        # ── Key Insights ──
        st.subheader("Key Insights")
        files = Config.DATASET_FILES.get(dataset) or []
        files_md = "\n".join(f"  - `{f}`" for f in files) if files else "  - All files in data/"
        st.info(f"""
**Dataset:** `{dataset}` → collection `{Config.get_collection_name(dataset)}`

**Source rows:** {f"{stats.get('source_row_count'):,}" if stats.get('source_row_count') else "re-ingest to compute"}  
**Indexed chunks:** {total_docs:,}

**Source files ({len(files)}):**
{files_md}

Governorate filtering is dynamic based on actual indexed data. Source citations are shown for every answer.
        """)

    except Exception as e:
        st.error(f"Failed to load statistics: {e}")

# Footer
st.markdown("---")
st.caption(
    "🇹🇳 Tunisia Open Government Data RAG | "
    "Multi-Dataset Support | Built with LangChain + Streamlit"
)