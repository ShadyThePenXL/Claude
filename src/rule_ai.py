from dataclasses import dataclass

from google import genai

MODEL = "gemini-2.5-flash"


@dataclass
class RuleCheckResult:
    passed: bool
    feedback: str


class RuleAI:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = system_prompt

    def _check(self, user_content: str) -> RuleCheckResult:
        system = self.system_prompt + (
            "\n\nRespond in this exact format:\n"
            "VERDICT: PASS or FAIL\n"
            "FEEDBACK: If FAIL, explain exactly what is wrong, which specific "
            "parts of the response need to change, and what the corrected "
            "version should do instead. Be detailed and actionable. "
            "If PASS, write 'None'."
        )
        response = self.client.models.generate_content(
            model=MODEL,
            contents=user_content,
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=512,
            ),
        )
        text = response.text
        passed = "VERDICT: PASS" in text.upper()
        feedback_line = ""
        for line in text.split("\n"):
            if line.strip().upper().startswith("FEEDBACK:"):
                feedback_line = line.split(":", 1)[1].strip()
                break
        if not feedback_line:
            feedback_line = text if not passed else ""
        return RuleCheckResult(passed=passed, feedback=feedback_line)

    def check_rules(self, player_action: str, narrative_response: str) -> RuleCheckResult:
        content = (
            f"[Player action]\n{player_action}\n\n"
            f"[Narrative response to check]\n{narrative_response}\n\n"
            "Only check the rules that are relevant to this specific action "
            "and response. Ignore rules that don't apply. If no rules are "
            "relevant or none are broken, pass it."
        )
        return self._check(content)

    def check_continuity(
        self, player_action: str, narrative_response: str, history: str
    ) -> RuleCheckResult:
        content = (
            f"[Game history]\n{history}\n\n"
            f"[Player action]\n{player_action}\n\n"
            f"[Narrative response to check]\n{narrative_response}\n\n"
            "Check for direct contradictions with established facts in the "
            "history (wrong location, dead characters appearing alive, items "
            "the player doesn't have). Minor details and creative additions "
            "that don't contradict anything are fine — pass those."
        )
        return self._check(content)

    def answer_question(self, question: str, history: str = "") -> str:
        context = ""
        if history:
            context = f"[Game history]\n{history}\n\n"
        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                f"{context}"
                f"[Player question]\n{question}\n\n"
                "Answer this question based on the game rules and history."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=1024,
            ),
        )
        return response.text

    def get_relevant_rules(self, player_action: str) -> str:
        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                f"[Player action]\n{player_action}\n\n"
                "Which rules apply to this action? List only the relevant "
                "rules as short bullet points. If none apply, say 'No specific "
                "rules apply.'"
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=512,
            ),
        )
        return response.text
