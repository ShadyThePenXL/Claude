from __future__ import annotations

from typing import TYPE_CHECKING

from google import genai

if TYPE_CHECKING:
    from .rule_ai import RuleAI
    from .history_ai import HistoryAI
    from .google_docs import GoogleDocsClient

MODEL_FULL = "gemini-2.5-flash"
MODEL_LITE = "gemini-2.5-flash-lite"

_DATA_INSTRUCTIONS = (
    "\n\nCRITICAL RULES FOR YOUR RESPONSE:"
    "\n1. The game document data below is ABSOLUTE TRUTH. Every name, "
    "stat, class, level, skill, location, date, and detail in the "
    "document is correct. You MUST match it exactly."
    "\n2. NEVER invent, guess, or change any facts. If the document "
    "says the player is in Deeproot Forest, they are in Deeproot "
    "Forest. If it says their class is Devourer, it is Devourer. "
    "If it says their Strength is 65, it is 65."
    "\n3. If the document doesn't have information you need, say so "
    "in the narrative rather than making something up."
    "\n4. For lookups (stats, attributes, inventory), present the "
    "exact data from the document."
    "\n5. Keep responses concise (2-4 paragraphs) unless the player "
    "asks for detail."
)


class NarrativeAI:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = system_prompt
        self.rule_ai: RuleAI | None = None
        self.history_ai: HistoryAI | None = None
        self.docs: GoogleDocsClient | None = None

    def _get_all_doc_data(self) -> str:
        if self.docs and self.docs.enabled:
            try:
                return self.docs.read_doc()
            except Exception:
                pass
        if self.history_ai:
            return self.history_ai.get_history()
        return ""

    def generate(
        self,
        player_action: str,
        context: str = "",
        feedback: str = "",
    ) -> str:
        doc_data = self._get_all_doc_data()

        user_content = ""

        if doc_data:
            user_content += (
                f"[GAME DOCUMENT — THIS IS THE SOURCE OF TRUTH. "
                f"USE THESE FACTS EXACTLY.]\n{doc_data}\n\n"
            )

        if self.rule_ai:
            rules = self.rule_ai.get_relevant_rules(player_action)
            if rules:
                user_content += f"[Rules that apply]\n{rules}\n\n"

        if feedback:
            user_content += (
                f"[Your previous draft was REJECTED. You MUST fix "
                f"these specific issues or you will be rejected again]\n"
                f"{feedback}\n\n"
            )

        user_content += f"[Player action]\n{player_action}"

        response = self.client.models.generate_content(
            model=MODEL_FULL,
            contents=user_content,
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt + _DATA_INSTRUCTIONS,
                max_output_tokens=20000,
            ),
        )
        return response.text or ""
