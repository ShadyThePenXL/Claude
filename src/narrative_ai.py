from __future__ import annotations

from typing import TYPE_CHECKING

from google import genai

if TYPE_CHECKING:
    from .rule_ai import RuleAI
    from .history_ai import HistoryAI
    from .google_docs import GoogleDocsClient

MODEL = "gemini-2.5-flash"


class NarrativeAI:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = system_prompt
        self.rule_ai: RuleAI | None = None
        self.history_ai: HistoryAI | None = None
        self.docs: GoogleDocsClient | None = None

    def _find_relevant_data(self, player_action: str) -> str:
        if not self.docs or not self.docs.enabled:
            if self.history_ai:
                history = self.history_ai.get_history()
                return history[-2000:] if history else ""
            return ""

        tabs = self.docs.get_tabs()
        if not tabs:
            return self.docs.read_doc()[:2000]

        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                f"[Available document tabs]\n{', '.join(tabs.keys())}\n\n"
                f"[Player action]\n{player_action}\n\n"
                "Which tabs contain information needed to respond to this "
                "action? List only the tab names, one per line. If you need "
                "multiple tabs, list them all."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction="You pick which document tabs are relevant. List tab names only.",
                max_output_tokens=100,
            ),
        )
        tab_picks = response.text or ""

        relevant_data = []
        for tab_name in tabs:
            if tab_name.lower() in tab_picks.lower():
                content = self.docs.read_tab_by_name(tab_name)
                if content.strip():
                    relevant_data.append(f"[{tab_name}]\n{content}")

        if not relevant_data:
            return self.docs.read_doc()[:2000]

        return "\n\n".join(relevant_data)

    def generate(
        self,
        player_action: str,
        context: str = "",
        feedback: str = "",
    ) -> str:
        user_content = ""

        if not feedback:
            doc_data = self._find_relevant_data(player_action)
            if doc_data:
                user_content += (
                    f"[Reference data from the game document — use this as "
                    f"your source of truth]\n{doc_data}\n\n"
                )

            if self.rule_ai:
                rules = self.rule_ai.get_relevant_rules(player_action)
                if rules:
                    user_content += f"[Rules that apply]\n{rules}\n\n"
        elif context:
            user_content += f"[Story context so far]\n{context}\n\n"

        if feedback:
            user_content += (
                f"[Your previous draft was rejected. Fix these issues]\n"
                f"{feedback}\n\n"
            )

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
                    "\n\nWhen reference data from the game document is provided, "
                    "treat it as the source of truth. Do not invent or change "
                    "any stats, items, names, locations, or facts — use exactly "
                    "what the document says."
                ),
                max_output_tokens=20000,
            ),
        )
        return response.text or ""
