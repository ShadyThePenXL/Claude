from __future__ import annotations

from typing import TYPE_CHECKING

from google import genai

if TYPE_CHECKING:
    from .rule_ai import RuleAI
    from .history_ai import HistoryAI

MODEL = "gemini-2.5-flash"


class NarrativeAI:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = system_prompt
        self.rule_ai: RuleAI | None = None
        self.history_ai: HistoryAI | None = None

    def _gather_info(self, player_action: str) -> str:
        info_parts = []

        if self.rule_ai:
            rules = self.rule_ai.get_relevant_rules(player_action)
            info_parts.append(f"[Rules that apply to this action]\n{rules}")

        if self.history_ai:
            history = self.history_ai.get_history()
            if history:
                info_parts.append(f"[Recent history]\n{history[-2000:]}")

        return "\n\n".join(info_parts)

    def generate(
        self,
        player_action: str,
        context: str = "",
        feedback: str = "",
    ) -> str:
        user_content = ""

        if not feedback and (self.rule_ai or self.history_ai):
            info = self._gather_info(player_action)
            if info:
                user_content += f"{info}\n\n"
        elif context:
            user_content += f"[Story context so far]\n{context}\n\n"

        if feedback:
            user_content += f"[Your previous draft was rejected. Fix these issues]\n{feedback}\n\n"

        user_content += f"[Player action]\n{player_action}"

        response = self.client.models.generate_content(
            model=MODEL,
            contents=user_content,
            config=genai.types.GenerateContentConfig(
                system_instruction=(
                    self.system_prompt
                    + "\n\nKeep responses concise (2-4 paragraphs) unless the "
                    "player explicitly asks for detail, a long description, "
                    "stats, lore, or similar. Match your length to what the "
                    "player's action calls for."
                ),
                max_output_tokens=20000,
            ),
        )
        return response.text or ""
