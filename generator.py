import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

class Generator:
    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0,
    ):
        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def generate(
        self,
        question: str,
        documents: list,
    ) -> str:

        context = "\n\n".join(
            [
                f"[Document {i + 1}]\n{doc['document']}"
                for i, doc in enumerate(documents)
            ]
        )

        prompt = f"""
You are a question-answering system.

Answer the question using only the provided context.

If the context does not contain enough information
to answer the question, say that you do not have
enough information.

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content