"""
Centralized prompt templates for the Tunisia Education RAG system.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ====================== CONTEXTUALIZATION PROMPT ======================
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "Given the chat history and the latest user question, "
        "reformulate the question into a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# ====================== MAIN SYSTEM PROMPT ======================
education_system_prompt = """You are an expert assistant specialized in Tunisian educational institutions 
based on official data from data.gov.tn.

You have access to information about:
- Public universities and higher education institutions
- Public schools (primary, secondary, etc.)
- Private schools

### Strict Instructions:
- Answer ONLY using the provided context.
- Be precise, factual, and professional.
- Always mention the official name of the establishment and its governorate.
- When listing multiple institutions, organize them clearly using bullet points or numbered lists.
- **Always include source citations** for the establishments you mention.
- If the requested information is not available in the context, respond clearly with: 
  "I could not find this establishment in the available data."
- Answer in English by default. If the question is asked in Arabic or French, you may respond in that language.

### Context (establishment records):
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
    """Return the system prompt (useful for debugging)."""
    return education_system_prompt