# AI Roleplay Game

A narrative RPG powered by a team of Gemini AI agents acting as your dungeon master. Play in the terminal or through a Streamlit web interface.

## How It Works

Four AI agents collaborate through a pipeline:

1. **Head Agent** -- Orchestrates the pipeline. Classifies player input (action, lookup, or meta) and routes it through the other agents. Does not read the game document directly.
2. **Narrative AI** -- Writes the story response based on your action, the full game document, and any applicable rules.
3. **Rule AI** -- Checks the narrative for two things: (1) did the AI speak for or make decisions for the player character, and (2) is the action impossible given the player's abilities (verified against the Player Status Sheet). If the action is impossible, the player can override and force it through.
4. **History AI** -- Extracts key events from each turn using terse, compressed notation. Minor updates write directly to a "History AI" tab in the Google Doc. Major updates go through Rule AI before writing to the main story tab.

### Turn Flow

```
Player action
  -> Head Agent classifies (ACTION / LOOKUP / META)
  -> Narrative AI drafts response (reads full game document)
  -> Rule AI checks response
     -> Spoke for player? FAIL, rewrite (up to 3 tries)
     -> Impossible action? Ask player to confirm or override
     -> PASS: deliver response
  -> History AI logs events
     -> MINOR: write directly to History AI tab
     -> MAJOR: check with Rule AI, then write to main tab
```

### Dual Model System

The game uses two Gemini models to balance quality and speed:

- **gemini-2.5-flash** (MODEL_FULL) -- Used for narrative generation, override responses, history commands, query answering, and head agent questions. Higher quality for player-facing content.
- **gemini-2.0-flash-lite** (MODEL_LITE) -- Used for classifiers (action type, event severity, tab picking), rule checks (yes/no verdicts), relevant rule extraction, terse history summaries, and player feedback summaries. Fast and cheap for internal decisions.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### 2. Gemini API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).

Set your API key in one of two ways:

- **Environment variable** (recommended):
  ```bash
  export GEMINI_API_KEY="your-api-key-here"
  ```
- **Config file**: Set `gemini_api_key` in `config.yaml`

### 3. Google Docs Integration (Optional)

If you want persistent history stored in a Google Doc:

1. Create a [Google Cloud service account](https://console.cloud.google.com/iam-admin/serviceaccounts) and enable the Google Docs API.
2. Download the service account JSON key file and save it as `credentials.json` in the project root.
3. Create a Google Doc and share it with the service account email address (as Editor).
4. Copy the document ID from the URL (`https://docs.google.com/document/d/<DOCUMENT_ID>/edit`) and set it in `config.yaml` under `google_docs.document_id`.
5. Create tabs in your Google Doc: a main story tab, a "History AI" tab for minor event logs, and a "Player Status Sheet" tab for character abilities and stats.

If you skip this, the game will use local in-memory history instead.

### 4. Configure Prompts

Edit `config.yaml` to customize:

- **Game title and description** -- shown in the terminal banner.
- **Narrative AI prompt** -- controls the writing style, tone, setting, and constraints.
- **Rule AI prompt** -- defines the rules the narrative must follow.
- **History AI prompt** -- controls how events are summarized for the log.

The template includes example prompts for a dark fantasy dungeon crawl. Replace them with your own setting.

## Running the Game

### Terminal Mode

```bash
python -m src.main
```

Or if installed as a package:

```bash
ai-rpg
```

You can also pass a custom config path:

```bash
python -m src.main path/to/my_config.yaml
```

Type your actions at the `>` prompt. Press `Ctrl+C` to quit.

### Web UI (Streamlit)

```bash
streamlit run src/app.py
```

This opens a ChatGPT-style chat interface in your browser. Player messages appear on the right, game responses on the left. Status updates (checking rules, generating narrative) are shown as spinners. All `!` commands work the same as in terminal mode.

## In-Game Commands

| Command | Description |
|---------|-------------|
| `!head <question>` | Ask the Head Agent a question about the game system |
| `!rules <question>` | Ask the Rule AI about game rules |
| `!narrative <prompt>` | Send a direct prompt to the Narrative AI |
| `!history <command>` | Process a history command (write to tabs, etc.) |

## Project Structure

```
├── config.yaml          # Game configuration and agent prompts
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Package metadata
├── src/
│   ├── __init__.py
│   ├── main.py          # Terminal UI entry point
│   ├── app.py           # Streamlit web UI entry point
│   ├── head_agent.py    # Orchestrator agent
│   ├── narrative_ai.py  # Story generation agent
│   ├── rule_ai.py       # Rule checking agent
│   ├── history_ai.py    # History management + Google Docs
│   ├── config.py        # Config file loader
│   └── google_docs.py   # Google Docs API wrapper
└── .gitignore
```
