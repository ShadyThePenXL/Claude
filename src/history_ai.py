import anthropic

from .google_docs import GoogleDocsClient

MODEL = "claude-sonnet-4-6"


class HistoryAI:
    def __init__(
        self,
        api_key: str,
        system_prompt: str,
        docs_client: GoogleDocsClient,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = system_prompt
        self.docs = docs_client
        self._local_history: list[str] = []

    def get_history(self) -> str:
        if self.docs.enabled:
            try:
                return self.docs.read_doc()
            except Exception:
                pass
        return "\n".join(self._local_history)

    def update_history(self, player_action: str, narrative_response: str) -> None:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"[Player action]\n{player_action}\n\n"
                        f"[Narrative response]\n{narrative_response}\n\n"
                        "Extract the key events as concise bullet points."
                    ),
                }
            ],
        )
        summary = response.content[0].text
        self._local_history.append(summary)

        if self.docs.enabled:
            try:
                self.docs.append_to_doc(summary)
            except Exception as e:
                print(f"\033[33m  [!] Failed to update Google Doc: {e}\033[0m")
