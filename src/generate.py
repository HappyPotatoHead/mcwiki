import ollama

from src import config


def generate_answer(query: str, context: str) -> str:
    prompt = f"""
You are a Minecraft knowledge assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer the question,
say that you do not have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"]
