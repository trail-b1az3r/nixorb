# Contributing to NixOrb

## Development Setup

```bash
git clone https://github.com/trail-b1az3r/nixorb.git
cd nixorb
bash scripts/setup_dev.sh
```

## Running Tests

```bash
bash scripts/run_tests.sh
# or directly:
python3 -m pytest tests/ -v
```

## Code Style

Uses **ruff** for linting and formatting:

```bash
ruff check nixorb/ tests/
ruff format nixorb/ tests/
```

Pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Writing a Plugin

Drop a `.py` file in `~/.local/share/nixorb/plugins/`:

```python
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What it does — the LLM reads this.",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input text"},
            },
            "required": ["input"],
        },
    },
}

def my_tool(input: str) -> str:
    return f"Result: {input}"
```

Async works too. Reload in Settings → Plugins → Reload.

## Architecture

See `docs/ARCHITECTURE.md` for the full threading and event-bus model.

## Pull Request Guidelines

1. Fork → feature branch → PR to `main`
2. All tests must pass
3. Ruff must report 0 errors
4. Add a CHANGELOG entry
5. Update docs if you change the config schema
