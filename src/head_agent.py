from .narrative_ai import NarrativeAI
from .rule_ai import RuleAI
from .history_ai import HistoryAI

MAX_RETRIES = 3

# ANSI color codes
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
    ):
        self.narrative = narrative
        self.rules = rules
        self.history = history
        self._turn_count = 0

    def process_action(self, player_action: str) -> str:
        if player_action.startswith("!"):
            command = player_action[1:].strip()
            _status("Processing command...")
            return self.history.process_command(command)

        self._turn_count += 1
        retries_used = 0
        history_context = ""

        _status("Gathering history context...")
        history_context = self.history.get_history()

        context_for_narrative = history_context[-2000:] if history_context else ""
        feedback = ""

        while retries_used < MAX_RETRIES:
            attempt = retries_used + 1
            if attempt > 1:
                _status(f"Rewriting narrative (attempt {attempt}/{MAX_RETRIES})...")
            else:
                _status("Generating narrative...")

            draft = self.narrative.generate(
                player_action=player_action,
                context=context_for_narrative,
                feedback=feedback,
            )

            _status("Checking rules...")
            rule_result = self.rules.check_rules(player_action, draft)

            if not rule_result.passed:
                retries_used += 1
                feedback = f"[Rule violation] {rule_result.feedback}"
                print(f"{YELLOW}  [!] Rule check failed: {rule_result.feedback}{RESET}")
                if retries_used >= MAX_RETRIES:
                    break
                continue

            _status("Verifying continuity...")
            if history_context:
                cont_result = self.rules.check_continuity(
                    player_action, draft, history_context
                )
                if not cont_result.passed:
                    retries_used += 1
                    feedback = f"[Continuity error] {cont_result.feedback}"
                    print(
                        f"{YELLOW}  [!] Continuity check failed: "
                        f"{cont_result.feedback}{RESET}"
                    )
                    if retries_used >= MAX_RETRIES:
                        break
                    continue

            _status("Updating history...")
            self.history.update_history(player_action, draft)
            return draft

        print(f"{RED}  [!] Max retries reached. Delivering best effort.{RESET}")
        self.history.update_history(player_action, draft)
        return draft
