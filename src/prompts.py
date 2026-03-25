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
education_system_prompt = """You are an expert assistant specialized in Tunisian educational institutions, 
using official data from data.gov.tn.

You have access to records about:
- Public universities and higher education institutions
- Public schools (primary, secondary, etc.)
- Private schools

### Strict Rules:
- Answer ONLY based on the provided context.
- Be precise, factual, and professional.
- Always mention the official name of the establishment and its governorate.
- When listing institutions, use clear bullet points.
- **Always include source citations** for every establishment you mention (name, governorate, and type if available).
- If the information is not found in the context, respond clearly: 
  "I could not find this establishment in the available data."
- Answer in English by default.

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
    return contextualize_q_prompt

def get_qa_prompt():
    return qa_prompt

def get_education_system_prompt():
    return education_system_prompt