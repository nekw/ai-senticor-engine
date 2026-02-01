Thank you for your interest in contributing!

Please follow these guidelines to make collaboration smooth:

- Fork the repo and create a feature branch for your change.
- Keep changes small and focused; write tests for new behavior.
- Run existing tests locally: `python -m pytest -q`.
- Use descriptive commit messages and open a pull request against `main`.

## Code Quality

**Pre-commit hooks**: Install and run before committing:
```bash
python -m pre_commit install
python -m pre_commit run --all-files
```

**Code style**:
- Follow PEP 8 for Python code.
- Use docstrings for public functions and classes.
- Pre-commit hooks enforce Black, isort, flake8, and mypy.

## Testing

**Run tests**:
```bash
pytest tests/ -v --cov=src --cov-report=term
```

**Security scanning**: CI automatically runs:
- Bandit (security linter)
- Safety (dependency vulnerabilities)
- pip-audit (CVE scanning)
- Trivy (container/filesystem scanning)

## Reporting Issues

- Include steps to reproduce and minimal repro data where possible.
- For security vulnerabilities, see [SECURITY.md](SECURITY.md).

Maintainers will review PRs and provide feedback. Thanks for helping improve this project!
