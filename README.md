# AI Roleplay Game

A narrative RPG powered by a team of Claude AI agents acting as your dungeon master. Type actions in the terminal and receive consistent, rule-checked story responses.

## How It Works

Four AI agents collaborate through a pipeline:

1. **Head Agent** -- Orchestrates the pipeline. Routes player input through the other agents and delivers the final response.
2. **Narrative AI** -- Writes the story response based on your action and the game's history.
3. **Rule AI** -- Checks the narrative against game rules and history for consistency. If it fails, the Narrative AI rewrites (up to 3 attempts).
4. **History AI** -- Extracts key events from each turn and logs them to a Google Doc (or local memory if Google Docs is not configured).

### Turn Flow

```
Player action
  -> Narrative AI drafts response
  -> Rule AI checks rules
     -> FAIL: Narrative AI rewrites (up to 3 tries total)
     -> PASS: Rule AI checks continuity against history
        -> FAIL: Narrative AI rewrites (shares the 3-try budget)
        -> PASS: Response delivered to player
  -> History AI logs the events
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

### 2. Anthropic API Key

Set your API key in one of two ways:

- **Environment variable** (recommended):
  ```bash
  export ANTHROPIC_API_KEY="sk-ant-..."
  ```
- **Config file**: Set `anthropic_api_key` in `config.yaml`

### 3. Google Docs Integration (Optional)

If you want persistent history stored in a Google Doc:

1. Create a [Google Cloud service account](https://console.cloud.google.com/iam-admin/serviceaccounts) and enable the Google Docs API.
2. Download the service account JSON key file and save it as `credentials.json` in the project root.
3. Create a Google Doc and share it with the service account email address (as Editor).
4. Copy the document ID from the URL (`https://docs.google.com/document/d/<DOCUMENT_ID>/edit`) and set it in `config.yaml` under `google_docs.document_id`.

If you skip this, the game will use local in-memory history instead.

### 4. Configure Prompts

Edit `config.yaml` to customize:

- **Game title and description** -- shown in the terminal banner.
- **Narrative AI prompt** -- controls the writing style, tone, setting, and constraints.
- **Rule AI prompt** -- defines the game rules that the narrative must follow.
- **History AI prompt** -- controls how events are summarized for the log.

The template includes example prompts for a dark fantasy dungeon crawl. Replace them with your own setting.

## Running the Game

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

## Project Structure

```
├── config.yaml          # Game configuration and agent prompts
├── requirements.txt     # Python dependencies
├── pyproject.toml       # Package metadata
├── src/
│   ├── __init__.py
│   ├── main.py          # Entry point and terminal UI
│   ├── head_agent.py    # Orchestrator agent
│   ├── narrative_ai.py  # Story generation agent
│   ├── rule_ai.py       # Rule and continuity checking agent
│   ├── history_ai.py    # History management + Google Docs
│   ├── config.py        # Config file loader
│   └── google_docs.py   # Google Docs API wrapper
└── .gitignore
```

## Model

All agents use `claude-sonnet-4-6` for a good balance of quality and cost.
