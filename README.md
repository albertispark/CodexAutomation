# Financial Data Pipeline

A local-first pipeline for extracting financial data from PDF, image, Excel,
and CSV files. OCR, indexing, retrieval, extraction, and redaction run locally
through Ollama. Only the compact redacted payload—without source snippets—is
eligible for cloud processing. Claude creates the analysis, then an independent
OpenAI model recomputes and peer-reviews that structured answer before the
pipeline can write a workbook. The final deliverable includes a formatted
workbook and local audit trail.

## Prerequisites

- macOS with Python 3.11 or newer
- Ollama running locally
- The configured local models:

```sh
ollama pull gemma4:e4b
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

If `gemma4:e4b` is unavailable in your registry, configure another
vision-capable model such as `gemma3:4b` in `config/settings.yaml`.

On a 16 GB Mac, enforce one loaded model in the Ollama server environment:

```sh
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
```

Restart Ollama.app afterward. Putting this setting in the project `.env` does
not affect the Ollama server.

## Install

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env.local
pipeline doctor
```

With `uv`, the equivalent editable install is `uv pip install -e '.[dev]'`.

Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in `.env.local`. The file is
gitignored and takes priority over a shared `.env`; existing shell environment
variables still take highest priority. Never paste keys into YAML or commit
them. `--no-cloud` and `--dry-run` work without either API key.

Peer review is enabled in `config/settings.yaml` and currently uses
`gpt-5.6-sol` with medium reasoning. Set `review.enabled: false` to run the
legacy Claude-only path. When enabled, a review must be approved or corrected
before Excel output is allowed; a rejection is quarantined. Review records are
written under `outputs/reviews/`, and identical review requests are cached
under `cache/review/`.

## Usage

```sh
pipeline run                         # process files under inputs/
pipeline run report.pdf
pipeline run report.pdf --dry-run   # local payload + local cost preview
pipeline run report.pdf --no-cloud  # local redacted payload JSON
pipeline run inputs/ --tasks metrics,variance
pipeline run inputs/ --batch
pipeline batch-poll msgbatch_...
pipeline index report.pdf
pipeline doctor
```

When `redaction.enabled` is false, `pipeline run` requires the explicit
`--allow-unredacted` consent flag. OpenAI receives only the redacted payload and
Claude's structured analysis, not the original file, filename, or source
snippets; review requests also set `store=false`. Confirm your organization's
Anthropic and OpenAI data-retention posture before using cloud stages in
production.

The CLI token total and audit log include OpenAI review usage. The displayed
cost estimate remains the Anthropic budget-ledger amount; check the OpenAI
usage dashboard for reviewer charges rather than relying on a hardcoded price.

Exit codes are `0` for success, `1` for a file-level failure or quarantine,
and `2` for an environment/configuration failure.

## Tests

```sh
pytest -q
OLLAMA_ITESTS=1 pytest tests/test_integration_local.py -q -s
```

The normal suite never contacts Anthropic or OpenAI. The optional
`ANTHROPIC_ITESTS=1 pytest tests/test_claude_client.py -q` run makes only the
explicit `count_tokens` cache-floor check and requires `ANTHROPIC_API_KEY`.
