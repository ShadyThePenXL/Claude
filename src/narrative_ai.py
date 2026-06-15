from google import genai

MODEL = "gemini-2.5-flash"


class NarrativeAI:
    def __init__(self, api_key: str, system_prompt: str):
        self.client = genai.Client(api_key=api_key)
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

        response = self.client.models.generate_content(
            model=MODEL,
            contents=user_content,
            config=genai.types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=4096,
            ),
        )
        return response.text
