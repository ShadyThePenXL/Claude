from __future__ import annotations

import re
from typing import TYPE_CHECKING

from google import genai

from .google_docs import GoogleDocsClient

if TYPE_CHECKING:
    from .rule_ai import RuleAI

MODEL = "gemini-2.5-flash"

_NO_MARKDOWN = (
    "\n\nWrite plain text only. No markdown, no asterisks, no # headers. "
    "Do not add titles like 'Proposed Update' or 'History AI Output'. "
    "Just write the actual content."
)


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text.strip()


class HistoryAI:
    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        docs_client: GoogleDocsClient,
    ):
        self.client = genai.Client(api_key=api_key)
        self.system_prompt = system_prompt
        self.docs = docs_client
        self._local_history: list[str] = []
        self.rule_ai: RuleAI | None = None

    def get_history(self) -> str:
        if self.docs.enabled:
            try:
                return self.docs.read_doc()
            except Exception:
                pass
        return "\n".join(self._local_history)

    def _classify_event(self, summary: str) -> bool:
        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                f"[Event summary]\n{summary}\n\n"
                "Is this a major event? Major events include:\n"
                "- Character death or serious injury\n"
                "- Gaining or losing important items/abilities\n"
                "- Major plot developments or reveals\n"
                "- Location changes to new areas\n"
                "- New significant characters or creatures appearing\n"
                "- Changes to the world or game state\n\n"
                "Respond with only MAJOR or MINOR."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction="You classify game events. Respond with one word only.",
                max_output_tokens=10,
            ),
        )
        return "MAJOR" in response.text.upper()

    def update_history(self, player_action: str, narrative_response: str) -> str | None:
        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                f"[Player action]\n{player_action}\n\n"
                f"[Narrative response]\n{narrative_response}\n\n"
                "Extract the key events as concise bullet points."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt + _NO_MARKDOWN,
                max_output_tokens=512,
            ),
        )
        summary = _strip_markdown(response.text)

        if self.rule_ai and self._classify_event(summary):
            history = self.get_history()
            check = self.rule_ai.check_continuity(
                player_action, narrative_response, history
            )
            if not check.passed:
                return check.feedback

        self._local_history.append(summary)

        if self.docs.enabled:
            try:
                self.docs.append_to_doc(summary)
            except Exception as e:
                print(f"\033[33m  [!] Failed to update Google Doc: {e}\033[0m")

        return None

    def process_command(self, command: str) -> str:
        tabs = self.docs.get_tabs() if self.docs.enabled else {}
        history = self.get_history()

        tab_list = ", ".join(tabs.keys()) if tabs else "none"

        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                f"[Available document tabs]\n{tab_list}\n\n"
                f"[Full game history]\n{history}\n\n"
                f"[Command]\n{command}\n\n"
                "Based on the command and the game history, generate the content "
                "to write. Start your response with:\n"
                "TAB: <tab name>\n"
                "Then write the content below that line."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt + _NO_MARKDOWN,
                max_output_tokens=1024,
            ),
        )

        text = _strip_markdown(response.text)

        target_tab = None
        content = text
        for i, line in enumerate(text.split("\n")):
            if line.strip().upper().startswith("TAB:"):
                tab_name = line.split(":", 1)[1].strip()
                for title, tid in tabs.items():
                    if (
                        tab_name.lower() in title.lower()
                        or title.lower() in tab_name.lower()
                    ):
                        target_tab = (title, tid)
                        break
                content = "\n".join(text.split("\n")[i + 1 :]).strip()
                break

        if not content:
            content = text

        if target_tab and self.docs.enabled:
            try:
                self.docs.append_to_tab(target_tab[1], content)
                return f"Updated the '{target_tab[0]}' tab in the Google Doc."
            except Exception as e:
                return f"Failed to update Google Doc: {e}"
        elif self.docs.enabled:
            try:
                self.docs.append_to_doc(content)
                return "Updated the Google Doc."
            except Exception as e:
                return f"Failed to update Google Doc: {e}"

        self._local_history.append(content)
        return "Google Docs is not enabled. Saved to local history."
