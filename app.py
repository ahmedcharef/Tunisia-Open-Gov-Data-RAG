import streamlit as st
from dotenv import load_dotenv
import sys
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from config import Config

load_dotenv()

st.set_page_config(
    page_title="🇹🇳 Tunisia Education RAG",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎓 Tunisia Education RAG")
st.markdown("**Assistant intelligent sur les établissements scolaires et universitaires en Tunisie**")
st.caption("Données officielles data.gov.tn | RGPH & Ministère de l'Éducation")

# ====================== SESSION STATE ======================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    try:
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        st.session_state.vectorstore = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="tn_education_etablissements_2025",   # Must match ingest.py
        )
        st.success("✅ Base de données chargée avec succès", icon="✅")
    except Exception as e:
        st.error(f"❌ Impossible de charger la base de données: {e}")
        st.stop()

# ====================== LLM SETUP ======================
@st.cache_resource
def get_llm():
    if Config.LLM_PROVIDER == "openrouter":
        return ChatOpenAI(
            model=Config.OPENROUTER_MODEL,
            api_key=Config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.25,
            max_tokens=2048,
        )
    else:
        return ChatOllama(
            model=Config.OLLAMA_MODEL,
            temperature=0.25,
        )

llm = get_llm()

# ====================== PROMPTS ======================
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformule la question en une requête autonome en tenant compte de l'historique."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_system_prompt = """Tu es un assistant expert des établissements d'enseignement en Tunisie.

Tu as accès aux données officielles concernant :
- Universités et établissements d'enseignement supérieur publics
- Établissements scolaires publics (primaire, secondaire, etc.)
- Établissements scolaires privés

**Instructions :**
- Réponds de manière claire, précise et organisée.
- Utilise des listes ou tableaux quand il y a plusieurs établissements.
- Mentionne le gouvernorat, la délégation, le type et l'adresse quand disponible.
- Si tu donnes une liste, limite-la à 8–10 résultats maximum pour plus de lisibilité.
- Si l'information n'est pas disponible, dis-le clairement.
- Réponds en français (ou en arabe si la question est en arabe).

Contexte :
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ====================== CHAINS ======================
@st.cache_resource
def get_rag_chain():
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8}
    )
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

rag_chain = get_rag_chain()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    k_value = st.slider("Nombre de documents à récupérer (k)", min_value=4, max_value=20, value=8)
    
    st.markdown("### Filtres rapides")
    selected_gov = st.selectbox(
        "Filtrer par Gouvernorat",
        options=["Tous", "TUNIS", "SFAX", "SOUSSE", "ARIANA", "BEN AROUS", "MANOUBA", 
                 "NABEUL", "BIZERTE", "MONASTIR", "MAHDIA", "KAIROUAN", "GAFSA", "MEDENINE"],
        index=0
    )
    
    if st.button("🔄 Réinitialiser la conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption("Modèle : " + (Config.OPENROUTER_MODEL if Config.LLM_PROVIDER == "openrouter" else Config.OLLAMA_MODEL))

# ====================== MAIN CHAT INTERFACE ======================
# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Posez votre question sur les établissements en Tunisie..."):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Recherche dans la base de données..."):
            try:
                # Apply gouvernorat filter if selected
                search_kwargs = {"k": k_value}
                if selected_gov != "Tous":
                    search_kwargs["filter"] = {"gouvernorat": {"$eq": selected_gov}}
                
                # Temporarily override retriever for this query
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs=search_kwargs
                )
                
                # Rebuild chain with current filter
                history_aware = create_history_aware_retriever(llm, retriever, contextualize_prompt)
                qa_chain = create_stuff_documents_chain(llm, qa_prompt)
                current_chain = create_retrieval_chain(history_aware, qa_chain)

                response = current_chain.invoke({
                    "input": prompt,
                    "chat_history": [(m["role"], m["content"]) for m in st.session_state.chat_history[:-1]]
                })

                answer = response["answer"]
                st.markdown(answer)

                # Add to history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Erreur lors de la génération : {str(e)}")

# Footer
st.markdown("---")
st.caption("🇹🇳 Tunisia Open Data RAG | Données : data.gov.tn | Construit avec LangChain + Streamlit")