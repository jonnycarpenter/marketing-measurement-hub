# Contributing

Thanks for your interest in contributing to Marketing Measurement Hub!

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure credentials

## Code Style

- Python code follows PEP 8
- Use type hints where practical
- Docstrings for public functions

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Test locally with `streamlit run app.py`
4. Submit a PR with a description of changes

## Agent Development

When modifying agents:
- Agent prompts live in `agents/agent_prompts/`
- Tools are defined in `agents/tools.py`
- Test handoffs between agents thoroughly

## Questions?

Open an issue or reach out at [ketzeroconsulting.ai](https://ketzeroconsulting.ai)
