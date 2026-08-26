"""
Centralized prompt templates for the Tunisia Education RAG system.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ====================== CONTEXTUALIZATION PROMPT ======================
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "Given the chat history and the latest user question, "
        "reformulate the question into a standalone, clear question that can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ====================== MAIN SYSTEM PROMPT ======================
# src/prompts.py  (replace the education_system_prompt)

education_system_prompt = """You are a helpful assistant specialized in Tunisian open government data from data.gov.tn.

You have access to records about educational institutions, transport infrastructure, social programs, and statistics.

### Instructions:
- Try your best to answer using the provided context.
- Be precise and mention names, governorates, and relevant details when possible.
- If you find relevant information, summarize it clearly.
- If you cannot find exact information, say "I could not find exact information about this, but here is what I found:" and give the closest matches.
- Always include source citations when you mention specific records.
- Answer in the same language as the user's question (Arabic, French, or English).

Context:
{context}
"""

# ====================== FINAL QA PROMPT ======================
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", education_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ====================== HELPER FUNCTIONS ======================

def get_contextualize_prompt():
    """Return the contextualization prompt."""
    return contextualize_q_prompt


def get_qa_prompt():
    """Return the main QA prompt."""
    return qa_prompt


def get_education_system_prompt():
    """Return the system prompt (useful for debugging or testing)."""
    return education_system_prompt