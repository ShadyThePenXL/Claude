"""Streamlit web UI for the AI Roleplay Game."""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.google_docs import GoogleDocsClient
from src.narrative_ai import NarrativeAI
from src.rule_ai import RuleAI
from src.history_ai import HistoryAI
from src.head_agent import HeadAgent


def _init_game() -> HeadAgent:
    """Initialize all agents and return the HeadAgent."""
    config_path = sys.argv[-1] if len(sys.argv) > 1 and sys.argv[-1].endswith(".yaml") else "config.yaml"

    config = Config(config_path)

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

    return head


def main():
    st.set_page_config(page_title="AI Roleplay Game", layout="centered")

    st.title("AI Roleplay Game")

    # Initialize session state
    if "head" not in st.session_state:
        try:
            st.session_state.head = _init_game()
        except (FileNotFoundError, ValueError) as e:
            st.error(f"Configuration error: {e}")
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_override" not in st.session_state:
        st.session_state.pending_override = None

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle pending impossible-action override
    if st.session_state.pending_override is not None:
        override = st.session_state.pending_override
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, do it anyway"):
                st.session_state.pending_override = None
                head = st.session_state.head
                with st.spinner("Generating override response..."):
                    history_context = head.history.get_history()
                    draft = head.rules.generate_override_response(
                        override["action"], history_context,
                    )
                    head.history.update_history(override["action"], draft)
                st.session_state.messages.append({"role": "assistant", "content": draft})
                st.rerun()
        with col2:
            if st.button("No, try something else"):
                st.session_state.pending_override = None
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Alright, try a different action."}
                )
                st.rerun()
        st.stop()

    # Chat input
    if player_input := st.chat_input("What do you do?"):
        st.session_state.messages.append({"role": "user", "content": player_input})
        with st.chat_message("user"):
            st.markdown(player_input)

        head = st.session_state.head

        # Handle ! commands
        if player_input.startswith("!"):
            with st.chat_message("assistant"):
                with st.spinner("Processing command..."):
                    response = head.process_action(player_input)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            return

        # Normal action flow
        with st.chat_message("assistant"):
            with st.spinner("Classifying action..."):
                action_type = head._classify_action(player_input)

            if action_type == "lookup":
                with st.spinner("Looking up game data..."):
                    response = head._handle_lookup(player_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            if action_type == "meta":
                with st.spinner("Thinking..."):
                    response = head._ask_head(player_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            # Action flow with rule checking
            history_context = head.history.get_history()
            feedback = ""
            max_retries = 3

            for attempt in range(max_retries):
                if attempt > 0:
                    with st.spinner(f"Rewriting narrative (attempt {attempt + 1}/{max_retries})..."):
                        draft = head.narrative.generate(
                            player_action=player_input, feedback=feedback,
                        )
                else:
                    with st.spinner("Generating narrative..."):
                        draft = head.narrative.generate(
                            player_action=player_input, feedback=feedback,
                        )

                with st.spinner("Checking response..."):
                    rule_result = head.rules.check_response(player_input, draft)

                if not rule_result.passed:
                    if rule_result.is_action_impossible:
                        with st.spinner("Summarizing issue..."):
                            summary = head._summarize_for_player(rule_result.feedback)
                        msg = f"That action shouldn't be possible: {summary}\n\nDo you really want to do this?"
                        st.markdown(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                        st.session_state.pending_override = {"action": player_input}
                        return

                    feedback = f"[Rule violation] {rule_result.feedback}"
                    continue

                with st.spinner("Updating history..."):
                    issue = head.history.update_history(player_input, draft)

                if issue:
                    # History AI flagged something -- show the draft anyway with the note
                    note = f"(History AI note: {issue})"
                    st.markdown(draft)
                    st.caption(note)
                    st.session_state.messages.append({"role": "assistant", "content": draft})
                    return

                st.markdown(draft)
                st.session_state.messages.append({"role": "assistant", "content": draft})
                return

            # Exhausted retries
            msg = (
                f"Could not get a clean response after {max_retries} tries. "
                f"Last issue: {feedback}\n\n{draft}"
            )
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})


if __name__ == "__main__":
    main()
else:
    # When run via `streamlit run src/app.py`, this module is imported
    main()
