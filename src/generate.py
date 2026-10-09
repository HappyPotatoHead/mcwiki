import ollama

from .models import Model


class Generator(Model):
    def generate_answer(self, query: str, context: str) -> str:
        prompt = self._generate_prompt(query, context)
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def _get_model(
        self,
    ) -> None:
        ollama.chat(model=self.model_name, messages=[], keep_alive="-1m")

    def _generate_prompt(self, query: str, context: str) -> str:
        return f"""
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
