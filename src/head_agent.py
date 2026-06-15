from google import genai

from .narrative_ai import NarrativeAI
from .rule_ai import RuleAI
from .history_ai import HistoryAI

MAX_RETRIES = 3
MODEL = "gemini-2.5-flash"

DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def _status(msg: str) -> None:
    print(f"{DIM}  >> {msg}{RESET}")


class HeadAgent:
    def __init__(
        self,
        narrative: NarrativeAI,
        rules: RuleAI,
        history: HistoryAI,
        api_key: str = "",
    ):
        self.narrative = narrative
        self.rules = rules
        self.history = history
        self._turn_count = 0
        self._client = genai.Client(api_key=api_key) if api_key else None

        self.narrative.rule_ai = rules
        self.narrative.history_ai = history

    def _handle_command(self, command: str) -> str:
        lower = command.lower()

        if lower.startswith("rules "):
            question = command[6:].strip()
            _status("Asking Rule AI...")
            history = self.history.get_history()
            return self.rules.answer_question(question, history)

        if lower.startswith("narrative "):
            prompt = command[10:].strip()
            _status("Asking Narrative AI...")
            return self.narrative.generate(player_action=prompt)

        if lower.startswith("history "):
            cmd = command[8:].strip()
            _status("Processing history command...")
            return self.history.process_command(cmd)

        _status("Processing command...")
        return self.history.process_command(command)

    def process_action(self, player_action: str) -> str:
        if player_action.startswith("!"):
            return self._handle_command(player_action[1:].strip())

        self._turn_count += 1
        retries_used = 0

        _status("Gathering history context...")
        history_context = self.history.get_history()
        feedback = ""

        while retries_used < MAX_RETRIES:
            attempt = retries_used + 1
            if attempt > 1:
                _status(f"Rewriting narrative (attempt {attempt}/{MAX_RETRIES})...")
            else:
                _status("Generating narrative...")

            draft = self.narrative.generate(
                player_action=player_action,
                feedback=feedback,
            )

            _status("Checking rules...")
            rule_result = self.rules.check_rules(player_action, draft)

            if not rule_result.passed:
                retries_used += 1
                print(f"{YELLOW}  [!] Rule check failed: {rule_result.feedback}{RESET}")
                if retries_used >= MAX_RETRIES:
                    break
                feedback = f"[Rule violation] {rule_result.feedback}"
                continue

            _status("Verifying continuity...")
            if history_context:
                cont_result = self.rules.check_continuity(
                    player_action, draft, history_context
                )
                if not cont_result.passed:
                    retries_used += 1
                    print(
                        f"{YELLOW}  [!] Continuity check failed: "
                        f"{cont_result.feedback}{RESET}"
                    )
                    if retries_used >= MAX_RETRIES:
                        break
                    feedback = f"[Continuity error] {cont_result.feedback}"
                    continue

            _status("Updating history...")
            self.history.update_history(player_action, draft)
            return draft

        print(f"{RED}  [!] Max retries reached. Delivering best effort.{RESET}")
        self.history.update_history(player_action, draft)
        return draft
