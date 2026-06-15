from dataclasses import dataclass

from google import genai

MODEL = "gemini-2.5-flash"


@dataclass
class RuleCheckResult:
    passed: bool
    feedback: str
    is_action_impossible: bool = False


class RuleAI:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = system_prompt

    def _check(self, user_content: str) -> RuleCheckResult:
        system = self.system_prompt + (
            "\n\nRespond in this exact format (keep it short):\n"
            "VERDICT: PASS or FAIL\n"
            "ISSUE_TYPE: ACTION or NARRATIVE\n"
            "FEEDBACK: 1-3 sentences max. What's wrong, what to fix.\n"
            "\n"
            "ACTION = the player's action itself is impossible.\n"
            "NARRATIVE = the action is fine but the writing is wrong.\n"
            "If PASS, just write 'None' for FEEDBACK."
        )
        response = self.client.models.generate_content(
            model=MODEL,
            contents=user_content,
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=20000,
            ),
        )
        text = response.text or ""
        passed = "VERDICT: PASS" in text.upper()
        is_action = "ISSUE_TYPE: ACTION" in text.upper()

        feedback_line = ""
        for line in text.split("\n"):
            if line.strip().upper().startswith("FEEDBACK:"):
                feedback_line = line.split(":", 1)[1].strip()
                break
        if not feedback_line:
            feedback_line = text if not passed else ""

        return RuleCheckResult(
            passed=passed, feedback=feedback_line, is_action_impossible=is_action,
        )

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

    def generate_override_response(
        self, player_action: str, history: str = ""
    ) -> str:
        context = ""
        if history:
            context = f"[Game history]\n{history}\n\n"
        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                f"{context}"
                f"[Player action]\n{player_action}\n\n"
                "The player has confirmed they want to do this. "
                "Write a pure narrative response — story only, no "
                "meta-commentary, no rules analysis, no 'Player Action:' "
                "headers, no 'Memory Updates:', no bullet points about "
                "game state. Just write the scene as it happens. "
                "Let the action succeed. Make it dramatic and show "
                "consequences, but keep it as narrative prose."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=20000,
            ),
        )
        return response.text or ""

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
                max_output_tokens=20000,
            ),
        )
        return response.text or ""

    def get_relevant_rules(self, player_action: str) -> str:
        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                f"[Player action]\n{player_action}\n\n"
                "Which rules apply? Short bullet points only. "
                "If none, say 'None.'"
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=512,
            ),
        )
        return response.text or ""
