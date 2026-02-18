# Contributing

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -e .[dev]
```

## Development Workflow

1. Create a feature branch.
2. Add or update tests in `tests/`.
3. Run checks locally:

```bash
pytest
```

4. Open a pull request with a clear summary of:
- What changed
- Why it changed
- How it was validated

## Coding Notes

- Keep functions typed and documented.
- Preserve stage ordering behavior for categorical stages.
- Maintain sparse adjacency workflows for scalability.
