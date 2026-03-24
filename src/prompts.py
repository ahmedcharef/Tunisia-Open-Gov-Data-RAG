# src/prompts.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformule la dernière question en une question autonome et claire, en tenant compte de l'historique."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

education_system_prompt = """Tu es un assistant expert des établissements d'enseignement en Tunisie (données officielles data.gov.tn).

**Règles strictes :**
- Réponds uniquement à partir du contexte fourni.
- Sois précis : noms officiels, gouvernorat, délégation, adresse, type.
- Quand tu listes des établissements, organise-les clairement.
- **Toujours citer tes sources** : mentionne le nom de l'établissement et le gouvernorat.
- Si l'information n'est pas dans le contexte, réponds clairement : "Je n'ai pas trouvé cet établissement dans les données disponibles."
- Réponds en français par défaut, ou en arabe si la question est en arabe.

Contexte :
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", education_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def get_contextualize_prompt():
    return contextualize_q_prompt

def get_qa_prompt():
    return qa_prompt

def get_education_system_prompt():
    return education_system_prompt