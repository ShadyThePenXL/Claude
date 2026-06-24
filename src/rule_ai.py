from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import llm_client as genai

if TYPE_CHECKING:
    from .google_docs import GoogleDocsClient

MODEL_FULL = "kimi-k2.6"
MODEL_LITE = "kimi-k2.6"


@dataclass
class RuleCheckResult:
    passed: bool
    feedback: str
    is_action_impossible: bool = False


class RuleAI:
    def __init__(self, api_key: str, system_prompt: str, docs: GoogleDocsClient | None = None):
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = system_prompt
        self.docs = docs

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
            model=MODEL_LITE,
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

    def check_response(self, player_action: str, narrative_response: str) -> RuleCheckResult:
        """Check two things:
        1. Did the Narrative AI speak for, act for, or make decisions for the player character?
        2. Is this action crazy/impossible given the player's abilities?
        """
        # First check: did the narrative speak for the player?
        content = (
            f"[Player action]\n{player_action}\n\n"
            f"[Narrative response to check]\n{narrative_response}\n\n"
            "Check these two things ONLY:\n"
            "1. Did the narrative speak for, act for, or make decisions for the "
            "player character? The narrative should describe the world's response "
            "but never put words in the player's mouth or decide what the player "
            "does next.\n"
            "2. Is this action crazy or impossible? If it seems impossible, "
            "flag it as ACTION type so we can check the player's abilities.\n\n"
            "If neither issue applies, PASS it."
        )
        result = self._check(content)

        # If flagged as impossible action, check the Player Status Sheet
        if not result.passed and result.is_action_impossible and self.docs:
            try:
                status_sheet = self.docs.read_tab_by_name("Player Status Sheet")
                if status_sheet.strip():
                    verify_content = (
                        f"[Player Status Sheet]\n{status_sheet}\n\n"
                        f"[Player action]\n{player_action}\n\n"
                        f"[Narrative response]\n{narrative_response}\n\n"
                        "The action was flagged as impossible. Check the Player "
                        "Status Sheet above. Does the player have the abilities, "
                        "items, or stats to do this? If YES, PASS it. If NO, "
                        "FAIL with ISSUE_TYPE: ACTION and explain why."
                    )
                    result = self._check(verify_content)
            except Exception:
                pass

        return result

    def generate_override_response(
        self, player_action: str, history: str = ""
    ) -> str:
        context = ""
        if history:
            context = f"[Game history]\n{history}\n\n"
        response = self.client.models.generate_content(
            model=MODEL_FULL,
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
            model=MODEL_FULL,
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
            model=MODEL_LITE,
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
