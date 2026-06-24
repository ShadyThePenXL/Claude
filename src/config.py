import os
from pathlib import Path

import yaml


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path) as f:
            self._data = yaml.safe_load(f)

        self.game_title = self._data["game"]["title"]
        self.game_description = self._data["game"]["description"]

        self.api_key = (
            self._data.get("moonshot_api_key")
            or os.environ.get("MOONSHOT_API_KEY")
            or ""
        )
        if not self.api_key:
            raise ValueError(
                "Moonshot API key must be set in config.yaml (moonshot_api_key) "
                "or the MOONSHOT_API_KEY environment variable. "
                "Get one at https://platform.moonshot.ai"
            )

        docs_cfg = self._data.get("google_docs", {})
        self.google_doc_id = docs_cfg.get("document_id", "")
        self.google_credentials_file = docs_cfg.get("credentials_file", "credentials.json")

        prompts = self._data.get("prompts", {})
        self.narrative_prompt = prompts.get("narrative", {}).get("system", "")
        self.rules_prompt = prompts.get("rules", {}).get("system", "")
        self.history_prompt = prompts.get("history", {}).get("system", "")
