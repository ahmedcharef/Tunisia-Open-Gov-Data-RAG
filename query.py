#!/usr/bin/env python3
"""
Tunisia Open Government Data RAG - Query Interface
Supports OpenRouter (configurable model via .env) + Ollama fallback
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
# Old (broken in 1.0+)
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# New (works with LangChain 1.0+)
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import logging

from config import Config, logger

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
load_dotenv()

# ────────────────────────────────────────────────
# 1. Load vector store & retriever
# ────────────────────────────────────────────────
try:
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    logger.info(f"Embeddings loaded: {Config.EMBEDDING_MODEL}")

    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=Config.COLLECTION_NAME,
    )
    logger.info(f"Vector store loaded: {Config.CHROMA_PERSIST_DIR} / collection '{Config.COLLECTION_NAME}'")

except Exception as e:
    logger.error(f"Failed to load vector store or embeddings: {e}")
    print("Error: Could not load the vector database. Did you run ingest.py first?")
    sys.exit(1)

# Retriever (can be overridden via CLI --k)
retriever = vectorstore.as_retriever(
    search_type=Config.SEARCH_TYPE,
    search_kwargs={"k": Config.RETRIEVER_K}
)

# ────────────────────────────────────────────────
# 2. LLM initialization
# ────────────────────────────────────────────────
if Config.LLM_PROVIDER == "openrouter":
    if not Config.OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY missing → cannot use OpenRouter")
        sys.exit(1)

    llm = ChatOpenAI(
        model=Config.OPENROUTER_MODEL,
        api_key=Config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=Config.TEMPERATURE,
        max_tokens=Config.MAX_TOKENS,
        extra_headers={
            "HTTP-Referer": "https://github.com/ahmedcharef/Tunisia-Open-Gov-Data-RAG",
            "X-Title": "Tunisia Open Data RAG",
        },
    )
    logger.info(f"LLM: OpenRouter – {Config.OPENROUTER_MODEL} (temp={Config.TEMPERATURE})")

elif Config.LLM_PROVIDER == "ollama":
    llm = ChatOllama(
        model=Config.OLLAMA_MODEL,
        temperature=Config.TEMPERATURE,
        num_ctx=32768,          # generous context for longer docs
        base_url=Config.OLLAMA_BASE_URL or "http://localhost:11434",
    )
    logger.info(f"LLM: Ollama – {Config.OLLAMA_MODEL} (local)")

else:
    logger.error(f"Unsupported LLM_PROVIDER: {Config.LLM_PROVIDER}")
    sys.exit(1)

# ────────────────────────────────────────────────
# 3. Prompts
# ────────────────────────────────────────────────
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Étant donné l'historique de la conversation et la dernière question de l'utilisateur "
     "(en français ou en arabe), reformulez-la en une question autonome claire."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_system_prompt = """Vous êtes un assistant expert sur les établissements d'enseignement en Tunisie (données officielles data.gov.tn).

Vous avez accès à des listes d'établissements :
- Universités et établissements publics d'enseignement supérieur
- Établissements scolaires publics
- Établissements scolaires privés

Instructions :
- Répondez en français (ou arabe si la question l'est)
- Soyez précis : noms officiels, adresses, gouvernorats, types d'établissements
- Pour les listes : présentez-les de façon claire (ex: tableau markdown si >3 items)
- Si l'utilisateur demande une liste → essayez de filtrer par gouvernorat, ville, type si mentionné
- Mentionnez la source : "Selon les données data.gov.tn / Ministère de l'Éducation / Enseignement Supérieur"
- Si pas d'information : "Je n'ai pas cet établissement dans les données actuelles."

Contexte pertinent (extraits de fiches d'établissements) :
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ────────────────────────────────────────────────
# 4. Chains
# ────────────────────────────────────────────────
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
question_answer_chain   = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ────────────────────────────────────────────────
# 5. CLI + interactive loop
# ────────────────────────────────────────────────
def run_single_query(query: str, chat_history: List[Tuple[str, str]]) -> str:
    try:
        result = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        return result["answer"]
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return f"Erreur lors du traitement : {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Tunisia Open Data RAG - Query Interface")
    parser.add_argument("--query", type=str, help="Posez une seule question (mode non-interactif)")
    parser.add_argument("--k", type=int, help="Nombre de documents à récupérer (surcharge config)")
    args = parser.parse_args()

    # Optional: override retriever k from CLI
    if args.k is not None and args.k > 0:
        global retriever
        retriever = vectorstore.as_retriever(
            search_type=Config.SEARCH_TYPE,
            search_kwargs={"k": args.k}
        )
        logger.info(f"Retriever depth overriden to k={args.k}")

    chat_history: List[Tuple[str, str]] = []

    if args.query:
        # Single-shot mode
        print("Question :", args.query)
        answer = run_single_query(args.query, chat_history)
        print("\nRéponse :")
        print(answer)
        return

    # Interactive mode
    print("="*60)
    print("  Tunisia Open Government Data RAG")
    print("  Modèle :", Config.OPENROUTER_MODEL if Config.LLM_PROVIDER == "openrouter" else Config.OLLAMA_MODEL)
    print("  Tapez 'exit', 'quit' ou Ctrl+C pour quitter")
    print("  'clear' pour effacer l'historique de conversation")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("Question (fr / ar) > ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("\nAu revoir !\n")
                break

            if user_input.lower() == "clear":
                chat_history.clear()
                print("Historique effacé.\n")
                continue

            if not user_input:
                continue

            print("\nRéflexion en cours...\n")

            answer = run_single_query(user_input, chat_history)
            print(answer)
            print("-"*60 + "\n")

            # Update history (keep last 8 messages ≈ 4 turns)
            chat_history.append(("human", user_input))
            chat_history.append(("ai", answer))
            if len(chat_history) > 8:
                chat_history = chat_history[-8:]

        except KeyboardInterrupt:
            print("\n\nArrêt par l'utilisateur.\n")
            break
        except Exception as e:
            logger.exception("Unexpected error in interactive loop")
            print("Une erreur inattendue est survenue. Consultez les logs.\n")

if __name__ == "__main__":
    main()