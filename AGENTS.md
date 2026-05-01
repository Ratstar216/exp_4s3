# Repository Guidelines

## Project Structure & Module Organization

This repository is a small Python project managed with `uv`.

- `hello.py` contains the current executable entry point.
- `pyproject.toml` defines package metadata, Python version requirements, and dependencies.
- `uv.lock` pins resolved dependency versions; keep it committed when dependencies change.
- `.python-version` specifies Python `3.13`.
- `README.md` is present but currently empty.

As the project grows, prefer moving application code into a package directory such as `src/jikken_3/` and tests into `tests/`. Keep generated files, virtual environments, and build output out of version control.

## Build, Test, and Development Commands

- `uv sync`: create or update the local virtual environment from `pyproject.toml` and `uv.lock`.
- `uv run python hello.py`: run the current application entry point.
- `uv add <package>`: add a runtime dependency and update the lockfile.
- `uv remove <package>`: remove a dependency and update the lockfile.

There is no build step configured yet. If packaging is added later, document the exact command here.

## Coding Style & Naming Conventions

Use Python 3.13 syntax and standard PEP 8 style. Prefer 4-space indentation, descriptive function names, and small functions with clear responsibilities.

Use `snake_case` for modules, functions, and variables. Use `PascalCase` for classes and `UPPER_SNAKE_CASE` for constants. Keep scripts import-safe by putting executable behavior under:

```python
if __name__ == "__main__":
    main()
```

No formatter or linter is configured yet. If tools such as `ruff`, `black`, or `mypy` are added, include their commands in this guide and enforce them consistently.

## Testing Guidelines

No test framework is currently configured. For new behavior, add tests under `tests/` and prefer `pytest` unless the project adopts another framework.

Recommended naming:

- Test files: `tests/test_<module>.py`
- Test functions: `test_<behavior>()`

Once `pytest` is added, run tests with `uv run pytest`.

## Commit & Pull Request Guidelines

This repository has no commits yet, so there is no existing commit convention to follow. Use short, imperative commit messages such as `Add image loader` or `Document setup commands`.

Pull requests should include a concise summary, relevant issue links if any, and the commands used to verify the change. For visual or OpenCV-related work, include before/after images or screenshots when they help reviewers understand the result.

## Security & Configuration Tips

Do not commit `.venv`, local caches, generated build artifacts, or secrets. Keep dependency changes intentional and review `uv.lock` updates before submitting.
