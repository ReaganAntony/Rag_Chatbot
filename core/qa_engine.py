"""
Question-Answering Engine Module

This module handles generating answers using Google's Gemini LLM
based on retrieved context from the document store.
"""

import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI with the new API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
# Using gemini-2.5-flash - latest model with improved performance
MODEL_NAME = "gemini-2.5-flash"

# System prompt for RAG-based QA
SYSTEM_PROMPT = """You are a helpful assistant. Answer the question ONLY using the provided context. If the answer isn't there, say you don't know. Always cite the page numbers used."""


def generate_answer(question: str, context: str) -> str:
    """
    Generate an answer to a question using the provided context.

    Uses Gemini LLM with a strict system prompt that ensures:
    - Answers are based ONLY on the provided context
    - The model admits when it doesn't know
    - Page numbers are cited in the response

    Args:
        question: The user's question.
        context: Retrieved context chunks with source citations.

    Returns:
        Generated answer string with citations.
    """
    if not context or context.strip() == "":
        return "I don't have any relevant information to answer this question. Please make sure documents have been ingested into the system."

    # Construct the full prompt
    full_prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    try:
        # Generate response using Gemini with new API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt
        )

        # Extract and return the text response
        if response and response.text:
            return response.text.strip()
        else:
            return "I was unable to generate a response. Please try again."

    except Exception as e:
        return f"Error generating answer: {str(e)}"


def generate_answer_with_history(question: str, context: str, chat_history: list = None) -> str:
    """
    Generate an answer considering previous conversation history.

    Args:
        question: The user's current question.
        context: Retrieved context chunks with source citations.
        chat_history: List of previous Q&A pairs [{"question": "...", "answer": "..."}, ...]

    Returns:
        Generated answer string with citations.
    """
    if not context or context.strip() == "":
        return "I don't have any relevant information to answer this question. Please make sure documents have been ingested into the system."

    # Build conversation history string
    history_str = ""
    if chat_history:
        history_parts = []
        for turn in chat_history[-5:]:  # Keep last 5 turns for context
            history_parts.append(f"User: {turn.get('question', '')}")
            history_parts.append(f"Assistant: {turn.get('answer', '')}")
        history_str = "\n".join(history_parts)

    # Construct the full prompt with history
    full_prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

{"CONVERSATION HISTORY:" + chr(10) + history_str + chr(10) if history_str else ""}
CURRENT QUESTION: {question}

ANSWER:"""

    try:
        # Generate response using Gemini with new API
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt
        )

        if response and response.text:
            return response.text.strip()
        else:
            return "I was unable to generate a response. Please try again."

    except Exception as e:
        return f"Error generating answer: {str(e)}"
