# Contributing

Thank you for helping. Bug reports, clear descriptions of confusing steps, and small fixes are all welcome.

## Reporting a problem

Open an issue with the bug template. Include the version (`fluentfolio --version`), your operating system and the error code, if there is one. A diagnostic report helps most: run `fluentfolio doctor --report report.txt`, read it, and attach it.

**Never attach recordings or transcripts**, yours or anyone else's, and never paste your Hugging Face token. For a security problem, follow [SECURITY.md](SECURITY.md) instead.

## Changing the code

You need [uv](https://docs.astral.sh/uv/) and Git. In a clone of the repository:

```
uv sync --locked --extra cpu
uv run --locked python tools/check.py fast
```

`tools/check.py fast` runs the format check, the linters (ruff, the import rules and bandit), the type check and the fast tests. It must pass before a pull request. `tools/check.py full` also runs the slow tests, which need the models and a browser (Microsoft Edge on Windows, Playwright's Chromium on Linux). It also runs two tools you install yourself: osv-scanner and gitleaks. Every other tool, shellcheck included, comes with `uv sync`. The slow tests need more parts than `fluentfolio components install --all` fetches, openSMILE needs the `nc` extra, and one test needs a speech model the app does not offer ([tests/README.md](tests/README.md#markers-and-skips) explains it).

Please keep a change small and about one thing, with a test that fails without it. Write user-facing text in plain English for language learners: short sentences, no jargon. Fluentfolio measures; it never gives a band, and no text may present its output as an official IELTS score.

Test data must be public or synthetic: never a real person's recording.

By contributing, you agree that your contribution is licensed under the MIT licence of this project.
