import anthropic

MODEL = "claude-sonnet-4-6"


class NarrativeAI:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.system_prompt = system_prompt

    def generate(
        self,
        player_action: str,
        context: str = "",
        feedback: str = "",
    ) -> str:
        user_content = ""
        if context:
            user_content += f"[Story context so far]\n{context}\n\n"
        if feedback:
            user_content += (
                f"[Your previous draft was rejected. Fix these issues]\n{feedback}\n\n"
            )
        user_content += f"[Player action]\n{player_action}"

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text
