# Technical Report

This directory contains the living technical report for the project.

## Build

```bash
./scripts/build_report.sh
```

The generated PDF is written to `docs/report/build/technical_report.pdf`.

## Commit policy

The repository includes a local `pre-commit` hook in `.githooks/pre-commit` that reminds us to update the report whenever code or benchmark logic changes. To enable it locally:

```bash
git config core.hooksPath .githooks
```

If a commit truly does not need a report change, bypass the reminder with:

```bash
SKIP_REPORT_CHECK=1 git commit ...
```
