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

    def _build_context(self, player_action: str, history: str) -> str:
        response = self._client.models.generate_content(
            model=MODEL,
            contents=(
                f"[Full game history]\n{history}\n\n"
                f"[Player's current action]\n{player_action}\n\n"
                "Extract only the facts the Narrative AI needs to write "
                "this scene. Return short bullet points covering:\n"
                "- Current location\n"
                "- Relevant nearby characters/creatures\n"
                "- Items the player has\n"
                "- Any recent events that directly affect this action\n"
                "Skip anything not relevant to this specific action."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction="You are a context summarizer. Be brief and factual.",
                max_output_tokens=512,
            ),
        )
        return response.text

    def _build_detailed_feedback(
        self, player_action: str, draft: str, failure_feedback: str,
        failure_type: str, history_context: str,
    ) -> str:
        _status(f"Getting detailed fix instructions from Rule AI...")
        explanation = self.rules.explain_failure(
            player_action, draft, failure_feedback, history_context
        )

        _status("Gathering supporting info for Narrative AI...")
        relevant_rules = self.rules.get_relevant_rules(player_action)

        history_snippet = ""
        if history_context:
            history_snippet = self._build_context(player_action, history_context)

        return (
            f"[{failure_type}]\n"
            f"{failure_feedback}\n\n"
            f"[Detailed instructions from Rule AI on how to fix this]\n"
            f"{explanation}\n\n"
            f"[Rules that apply to this action]\n"
            f"{relevant_rules}\n\n"
            f"[Key facts from history you must respect]\n"
            f"{history_snippet}"
        )

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
            history = self.history.get_history()
            context = history[-2000:] if history else ""
            return self.narrative.generate(player_action=prompt, context=context)

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

        context_for_narrative = ""
        if history_context and self._client:
            _status("Summarizing context...")
            context_for_narrative = self._build_context(
                player_action, history_context
            )
        elif history_context:
            context_for_narrative = history_context[-2000:]
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
                print(f"{YELLOW}  [!] Rule check failed: {rule_result.feedback}{RESET}")
                if retries_used >= MAX_RETRIES:
                    break
                feedback = self._build_detailed_feedback(
                    player_action, draft, rule_result.feedback,
                    "Rule violation", history_context,
                )
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
                    feedback = self._build_detailed_feedback(
                        player_action, draft, cont_result.feedback,
                        "Continuity error", history_context,
                    )
                    continue

            _status("Updating history...")
            self.history.update_history(player_action, draft)
            return draft

        print(f"{RED}  [!] Max retries reached. Delivering best effort.{RESET}")
        self.history.update_history(player_action, draft)
        return draft
