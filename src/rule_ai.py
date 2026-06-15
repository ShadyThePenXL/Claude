from dataclasses import dataclass

import anthropic

MODEL = "claude-sonnet-4-6"


@dataclass
class RuleCheckResult:
    passed: bool
    feedback: str


class RuleAI:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = system_prompt

    def _check(self, user_content: str) -> RuleCheckResult:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=self.system_prompt + (
                "\n\nRespond in this exact format:\n"
                "VERDICT: PASS or FAIL\n"
                "FEEDBACK: <explanation of what rules were broken, or 'None' if passed>"
            ),
            messages=[{"role": "user", "content": user_content}],
        )
        text = response.content[0].text
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
            "Check this response against the game rules. "
            "Does it break any rules?"
        )
        return self._check(content)

    def check_continuity(
        self, player_action: str, narrative_response: str, history: str
    ) -> RuleCheckResult:
        content = (
            f"[Game history]\n{history}\n\n"
            f"[Player action]\n{player_action}\n\n"
            f"[Narrative response to check]\n{narrative_response}\n\n"
            "Check this response for continuity with the game history. "
            "Are there any contradictions, impossible events, or inconsistencies?"
        )
        return self._check(content)
