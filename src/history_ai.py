from google import genai

from .google_docs import GoogleDocsClient

MODEL = "gemini-2.5-flash"


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

    def get_history(self) -> str:
        if self.docs.enabled:
            try:
                return self.docs.read_doc()
            except Exception:
                pass
        return "\n".join(self._local_history)

    def update_history(self, player_action: str, narrative_response: str) -> None:
        response = self.client.models.generate_content(
            model=MODEL,
            contents=(
                f"[Player action]\n{player_action}\n\n"
                f"[Narrative response]\n{narrative_response}\n\n"
                "Extract the key events as concise bullet points."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=512,
            ),
        )
        summary = response.text
        self._local_history.append(summary)

        if self.docs.enabled:
            try:
                self.docs.append_to_doc(summary)
            except Exception as e:
                print(f"\033[33m  [!] Failed to update Google Doc: {e}\033[0m")
