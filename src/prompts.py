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
education_system_prompt = """You are an expert assistant specialized in Tunisian educational institutions 
using official data from data.gov.tn.

You have access to records about:
- Public universities and higher education institutions
- Public schools (primary, secondary, etc.)
- Private schools

### Strict Instructions:
- Answer ONLY using the information from the provided context.
- Be precise, factual, and professional.
- For every establishment you mention, **always include its name and governorate** directly in your answer.
- When listing multiple institutions, use clear bullet points.
- **Strongly include source citations** in the body of your response. 
  Example: "The National Engineering School of Tunis (ENIT) is located in Tunis governorate (source: official education records)."
- If the requested information is not found in the context, respond clearly with: 
  "I could not find this establishment in the available data."
- Answer in English by default. If the question is in French or Arabic, you may respond in that language.

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
    """Return the system prompt (useful for debugging or testing)."""
    return education_system_prompt