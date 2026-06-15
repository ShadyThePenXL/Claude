from google import genai

from .narrative_ai import NarrativeAI
from .rule_ai import RuleAI
from .history_ai import HistoryAI
from .google_docs import GoogleDocsClient

MAX_RETRIES = 3
MODEL = "gemini-2.5-flash"

DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _status(msg: str) -> None:
    print(f"{DIM}  >> {msg}{RESET}")


class HeadAgent:
    def __init__(
        self,
        narrative: NarrativeAI,
        rules: RuleAI,
        history: HistoryAI,
        docs: GoogleDocsClient,
        api_key: str = "",
    ):
        self.narrative = narrative
        self.rules = rules
        self.history = history
        self.docs = docs
        self._turn_count = 0
        self._client = genai.Client(api_key=api_key) if api_key else None

        self.narrative.rule_ai = rules
        self.narrative.history_ai = history
        self.narrative.docs = docs
        self.history.rule_ai = rules

    def _classify_action(self, player_action: str) -> str:
        response = self._client.models.generate_content(
            model=MODEL,
            contents=(
                f"[Player input]\n{player_action}\n\n"
                "Classify this input as one of:\n"
                "LOOKUP — player wants to see data (attributes, stats, "
                "inventory, character sheet, map, history, lore, etc.)\n"
                "ACTION — player is doing something in the game world\n"
                "META — player is talking about the game system itself\n\n"
                "Respond with one word only."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction="You classify player inputs. Respond with one word only.",
                max_output_tokens=10,
            ),
        )
        text = response.text or ""
        text = text.strip().upper()
        if "LOOKUP" in text:
            return "lookup"
        if "META" in text:
            return "meta"
        return "action"

    def _ask_head(self, question: str) -> str:
        history = self.history.get_history()
        context = ""
        if history:
            context = f"[Game history]\n{history[-2000:]}\n\n"
        response = self._client.models.generate_content(
            model=MODEL,
            contents=(
                f"{context}"
                f"[Player question]\n{question}\n\n"
                "You are the head agent overseeing a multi-agent RPG system. "
                "You coordinate the Narrative AI (writes story), Rule AI "
                "(enforces rules), and History AI (tracks events). "
                "Answer the player's question or help them with whatever "
                "they need."
            ),
            config=genai.types.GenerateContentConfig(
                system_instruction=(
                    "You are the head orchestrator of an AI-powered RPG. "
                    "Be helpful and direct. You know how the whole system works."
                ),
                max_output_tokens=2048,
            ),
        )
        return response.text or ""

    def _handle_command(self, command: str) -> str:
        lower = command.lower()

        if lower.startswith("head "):
            question = command[5:].strip()
            _status("Thinking...")
            return self._ask_head(question)

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

    def _handle_lookup(self, player_action: str) -> str:
        _status("Looking up game data...")
        answer = self.history.answer_query(player_action)
        if answer:
            return answer
        _status("No data found, generating response...")
        return self.narrative.generate(player_action=player_action)

    def _ask_player_about_issue(self, draft: str, issue: str) -> str:
        print()
        print(f"{CYAN}{BOLD}  [Head Agent]{RESET}")
        print(f"{CYAN}  The History AI flagged a potential issue with this response:{RESET}")
        print(f"{YELLOW}  {issue}{RESET}")
        print()
        print(f"{CYAN}  What would you like to do?{RESET}")
        print(f"{DIM}  1) Accept it anyway{RESET}")
        print(f"{DIM}  2) Rewrite it{RESET}")
        print(f"{DIM}  3) Tell me what to change{RESET}")
        print()

        try:
            choice = input(f"{GREEN}{BOLD}  >> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            return "accept"

        if choice == "1" or choice.lower().startswith("a"):
            return "accept"
        elif choice == "2" or choice.lower().startswith("r"):
            return "rewrite"
        else:
            return choice

    def process_action(self, player_action: str) -> str:
        if player_action.startswith("!"):
            return self._handle_command(player_action[1:].strip())

        self._turn_count += 1

        _status("Classifying action...")
        action_type = self._classify_action(player_action)

        if action_type == "lookup":
            return self._handle_lookup(player_action)

        if action_type == "meta":
            return self._ask_head(player_action)

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
            issue = self.history.update_history(player_action, draft)

            if issue:
                player_response = self._ask_player_about_issue(draft, issue)

                if player_response == "accept":
                    self.history._local_history.append(draft)
                    if self.history.docs.enabled:
                        try:
                            self.history.docs.append_to_doc(draft)
                        except Exception:
                            pass
                    return draft
                elif player_response == "rewrite":
                    retries_used += 1
                    feedback = f"[History issue] {issue}"
                    if retries_used >= MAX_RETRIES:
                        break
                    continue
                else:
                    retries_used += 1
                    feedback = (
                        f"[History issue] {issue}\n"
                        f"[Player instructions] {player_response}"
                    )
                    if retries_used >= MAX_RETRIES:
                        break
                    continue

            return draft

        print(f"{RED}  [!] Max retries reached. Delivering best effort.{RESET}")
        self.history.update_history(player_action, draft)
        return draft
