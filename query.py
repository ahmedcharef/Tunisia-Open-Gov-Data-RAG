#!/usr/bin/env python3
"""
query.py - Tunisian Education Establishments RAG
Improved version with metadata filtering and better prompting
"""

import argparse
import sys
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from config import Config, logger

import warnings
import logging

# Silence only the specific multilingual-e5-large warning
warnings.filterwarnings(
    "ignore", 
    message=".*position_ids.*UNEXPECTED.*"
)

# Reduce verbosity from sentence-transformers and transformers
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

# ====================== GLOBAL VARIABLES ======================
COLLECTION_NAME = "tn_education_etablissements_2025"

# ====================== LOAD VECTOR STORE ======================
try:
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    logger.info(f"✅ Vector store loaded | Collection: {COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to load vector store: {e}")
    print("❌ Could not load the database. Run `python ingest.py` first.")
    sys.exit(1)

# ====================== LLM SETUP ======================
if Config.LLM_PROVIDER == "openrouter":
    llm = ChatOpenAI(
        model=Config.OPENROUTER_MODEL,
        api_key=Config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        model_kwargs={
            "extra_headers": {
                "HTTP-Referer": "https://github.com/ahmedcharef/Tunisia-Open-Gov-Data-RAG",
                "X-Title": "Tunisia Open Data RAG",
            }
        },
    )
    logger.info(f"Using OpenRouter → {Config.OPENROUTER_MODEL}")

elif Config.LLM_PROVIDER == "ollama":
    llm = ChatOllama(
        model=Config.OLLAMA_MODEL,
        temperature=Config.TEMPERATURE,
        num_ctx=32768,
    )
    logger.info(f"Using Ollama → {Config.OLLAMA_MODEL}")

else:
    logger.error(f"Unsupported LLM_PROVIDER: {Config.LLM_PROVIDER}")
    sys.exit(1)
# ====================== PROMPTS ======================
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformule la dernière question en une requête autonome en tenant compte de l'historique."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_system_prompt = """Tu es un assistant expert des établissements d'enseignement en Tunisie (données officielles data.gov.tn).

Tu connais :
- Les universités et établissements publics d'enseignement supérieur
- Les établissements scolaires publics
- Les établissements scolaires privés

Réponds de façon claire et structurée.
Utilise les noms officiels, gouvernorats, délégations et adresses quand disponibles.
Si tu donnes plusieurs établissements, présente-les sous forme de liste claire.
Si l'information n'est pas dans le contexte, dis : "Je n'ai pas trouvé cet établissement dans les données disponibles."

Contexte :
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ====================== RETRIEVER FUNCTION ======================
def get_retriever(k: int = 8, gouvernorat: str = None):
    """Create retriever with optional gouvernorat filter"""
    search_kwargs = {"k": k}
    if gouvernorat:
        search_kwargs["filter"] = {"gouvernorat": gouvernorat.upper()}

    return vectorstore.as_retriever(
        search_type="mmr",           # Better diversity
        search_kwargs=search_kwargs
    )

# ====================== CHAINS (created once) ======================
retriever = get_retriever(k=Config.RETRIEVER_K)
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ====================== HELPER ======================
def extract_gouvernorat(query: str) -> str | None:
    """Simple detection of governorate in query"""
    query_lower = query.lower()
    gouvernorats = {
        "tunis", "sfax", "sousse", "ariana", "ben arous", "manouba", "nabeul", "bizerte",
        "béja", "jendouba", "kairouan", "kasserine", "gafsa", "medenine", "gabes",
        "kebili", "tataouine", "zaghouan", "siliana", "mahdia", "monastir", "tozeur"
    }
    for gov in gouvernorats:
        if gov in query_lower:
            return gov.upper()
    return None

# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser(description="Tunisia Education RAG")
    parser.add_argument("--query", type=str, help="Single query")
    parser.add_argument("--k", type=int, default=Config.RETRIEVER_K, help="Number of results")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    args = parser.parse_args()

    if args.stats:
        count = vectorstore._collection.count()
        print(f"📊 Total establishments in database: {count:,}")
        return

    chat_history: List[Tuple[str, str]] = []

    if args.query:
        gov = extract_gouvernorat(args.query)
        local_retriever = get_retriever(k=args.k, gouvernorat=gov)
        local_chain = create_retrieval_chain(
            create_history_aware_retriever(llm, local_retriever, contextualize_prompt),
            question_answer_chain
        )
        response = local_chain.invoke({"input": args.query, "chat_history": chat_history})
        print("\n🤖 Réponse:\n", response["answer"])
        return

    # === Interactive Mode ===
    print("="*75)
    print("   🇹🇳 Tunisia Education RAG - Établissements scolaires & universitaires")
    print("="*75)

    while True:
        try:
            user_input = input("\nQuestion (fr/ar) > ").strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Au revoir !")
                break
            if user_input.lower() == "clear":
                chat_history.clear()
                print("🧹 Historique effacé.")
                continue
            if not user_input:
                continue

            # Auto-detect governorate for filtering
            gov = extract_gouvernorat(user_input)
            current_retriever = get_retriever(k=args.k, gouvernorat=gov)

            # Rebuild chain with current retriever
            current_history_retriever = create_history_aware_retriever(
                llm, current_retriever, contextualize_prompt
            )
            current_rag_chain = create_retrieval_chain(
                current_history_retriever, question_answer_chain
            )

            response = current_rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            answer = response["answer"]
            print(f"\n🤖 Réponse:\n{answer}\n")

            # Update history
            chat_history.extend([("human", user_input), ("ai", answer)])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            print("\n\nArrêté par l'utilisateur.")
            break
        except Exception as e:
            logger.error(f"Erreur: {e}")
            print("❌ Une erreur est survenue.")

if __name__ == "__main__":
    main()