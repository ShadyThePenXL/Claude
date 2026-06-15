import sys

from .config import Config
from .google_docs import GoogleDocsClient
from .narrative_ai import NarrativeAI
from .rule_ai import RuleAI
from .history_ai import HistoryAI
from .head_agent import HeadAgent

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

BANNER = r"""
 ___  ____    ____        _            _
|_ _||  _ \  |  _ \  ___ | | ___ _ __ | | __ _ _   _
 | | | |_) | | |_) |/ _ \| |/ _ \ '_ \| |/ _` | | | |
 | | |  __/  |  _ <| (_) | |  __/ |_) | | (_| | |_| |
|___||_|     |_| \_\\___/|_|\___| .__/|_|\__,_|\__, |
                                |_|            |___/
"""


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    try:
        config = Config(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\033[31mError: {e}\033[0m")
        sys.exit(1)

    docs_client = GoogleDocsClient(
        credentials_file=config.google_credentials_file,
        document_id=config.google_doc_id,
    )

    narrative = NarrativeAI(api_key=config.api_key, system_prompt=config.narrative_prompt)
    rules = RuleAI(api_key=config.api_key, system_prompt=config.rules_prompt, docs=docs_client)
    history = HistoryAI(
        api_key=config.api_key,
        system_prompt=config.history_prompt,
        docs_client=docs_client,
    )
    head = HeadAgent(
        narrative=narrative, rules=rules, history=history,
        api_key=config.api_key,
    )

    print(f"{CYAN}{BANNER}{RESET}")
    print(f"{BOLD}  {config.game_title}{RESET}")
    print(f"{DIM}  {config.game_description}{RESET}")
    print()

    if docs_client.enabled:
        print(f"{DIM}  Google Docs history: enabled{RESET}")
    else:
        print(f"{DIM}  Google Docs history: disabled (using local memory){RESET}")

    print(f"{DIM}  Type your actions below. Press Ctrl+C to quit.{RESET}")
    print(f"{DIM}  Commands: !head  |  !rules  |  !narrative  |  !history{RESET}")
    print(f"{DIM}  {'=' * 60}{RESET}")
    print()

    try:
        while True:
            try:
                action = input(f"{GREEN}{BOLD}> {RESET}").strip()
            except EOFError:
                break

            if not action:
                continue

            print()
            response = head.process_action(action)
            if response:
                print()
                print(f"{MAGENTA}{response}{RESET}")
                print()

    except KeyboardInterrupt:
        print(f"\n\n{DIM}  Farewell, adventurer.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
