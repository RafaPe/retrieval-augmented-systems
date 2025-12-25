"""
TRUE score = (# supported claims) / (total claims)
"""

from langchain_community.llms import Ollama


class TRUEEvaluator:
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.llm = Ollama(model=model, base_url=base_url)

    def extract_claims(self, answer: str) -> list[str]:
        prompt = f"""
            Extract factual claims from the answer below.

            Rules:
            - One claim per line
            - NO explanations
            - NO numbering
            - NO bullets
            - Output ONLY the claims
            - Lines strart with "- "

            Answer:
            {answer}
        """
        response = self.llm.invoke(prompt)

        # print(response)

        claims = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                claims.append(line.lstrip("- ").strip())

        return claims

    def check_claim(self, claim: str, context: str) -> bool:
        """
        Checks whether a claim is fully supported by the context.
        """
        prompt = f"""
            Context:
            {context}

            Claim:
            {claim}

            Is the claim fully supported by the context?
            Answer ONLY with "yes" or "no".
        """
        verdict = self.llm.invoke(prompt).lower()
        return "yes" in verdict

    def score(self, answer: str, context: str) -> float:
        claims = self.extract_claims(answer)

        if not claims:
            print("No claims found in the answer.")
            return 1.0  # No claims = no hallucination

        supported = 0
        for claim in claims:
            if self.check_claim(claim, context):
                supported += 1

        return supported / len(claims)


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    context = """
Bayesian inference is a method of statistical inference
that uses Bayes' theorem to update probabilities.
"""

    answer = """
Bayesian inference uses Bayes' theorem and was developed in the 18th century.
"""

    evaluator = TRUEEvaluator()
    score = evaluator.score(answer, context)

    print("TRUE score:", score)
