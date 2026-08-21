# import os

# from langchain_groq import ChatGroq
# from langchain_core.messages import HumanMessage
# from dotenv import load_dotenv
# load_dotenv()

# class LLM:
#     """
#     Thin LLM abstraction shared by all six actions.

#     Keeps ChatGroq/Groq specifics out of the action layer so the
#     executors only ever call `llm.complete(prompt)`.
#     """

#     def __init__(
#         self,
#         model: str = "openai/gpt-oss-20b",
#         temperature: float = 0,
#     ):
#         self.llm = ChatGroq(
#             model=model,
#             temperature=temperature,
#             api_key=os.getenv("GROQ_API_KEY"),
#         )

#     def complete(self, prompt: str) -> str:
#         response = self.llm.invoke(
#             [HumanMessage(content=prompt)]
#         )

#         return response.content

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


class LLM:
    """
    Thin LLM abstraction shared by all six actions.

    Keeps Ollama specifics out of the action layer so the
    executors only ever call `llm.complete(prompt)`.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0,
    ):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )

    def complete(self, prompt: str) -> str:
        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        return response.content