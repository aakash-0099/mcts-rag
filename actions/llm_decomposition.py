from .decomposition import DecompositionStrategy


class LLMDecomposition(DecompositionStrategy):

    def __init__(self, llm):
        self.llm = llm

    def decompose(self, question: str):
        if not isinstance(question, str):
            raise TypeError("question must be a string")

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty")

        prompt = f"""
            Decompose the following question into a sequence of dependent
            sub-questions required to answer it.

            Rules:
            - Return one sub-question per line.
            - Each line must represent exactly one retrieval/reasoning hop.
            - The first hop must be answerable directly from the original question.
            - Every later hop MUST use the answer from the previous relevant hop.
            - Represent dependencies explicitly using placeholders:
            <ANSWER_1>, <ANSWER_2>, <ANSWER_3>, etc.
            - NEVER use vague references such as "that person", "the supervisor",
            "they", "there", or "this equipment".
            - Replace those references with the appropriate <ANSWER_i> placeholder.
            - The final hop should ask for the actual answer requested by the original question.
            - Do not answer any sub-question.
            - Do not add explanations.
            - Do not number the lines.

            Example:

            Question:
            Who founded Company X, where was that person born, and what school
            did they attend?

            Output:
            Who founded Company X?
            Where was <ANSWER_1> born?
            What school did <ANSWER_1> attend?

            Question:
            {question}

            Sub-questions:
        """

        response = self.llm.complete(prompt).strip()

        subquestions = [
            line.strip()
            for line in response.splitlines()
            if line.strip()
        ]

        cleaned = []

        for subquestion in subquestions:
            if len(subquestion) >= 3 and subquestion[0].isdigit():
                parts = subquestion.split(". ", 1)
                if len(parts) == 2:
                    subquestion = parts[1]

            cleaned.append(subquestion.strip())

        return cleaned or [question]