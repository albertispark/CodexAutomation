# Financial Data Pipeline

A local-first pipeline for extracting financial data from PDF, image, Excel,
and CSV files. OCR, indexing, retrieval, extraction, and redaction run locally
through Ollama. Only the compact redacted payload—without source snippets—is
eligible for Claude analysis. The final deliverable is a formatted workbook
with a local audit trail.

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
cp .env.example .env
pipeline doctor
```

With `uv`, the equivalent editable install is `uv pip install -e '.[dev]'`.

Set `ANTHROPIC_API_KEY` in `.env` for the cloud stage. `--no-cloud` and
`--dry-run` work without an API key.

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
`--allow-unredacted` consent flag. Confirm your organization's Anthropic data
retention posture, including any zero-data-retention agreement, before using
the cloud stage in production.

Exit codes are `0` for success, `1` for a file-level failure or quarantine,
and `2` for an environment/configuration failure.

## Tests

```sh
pytest -q
OLLAMA_ITESTS=1 pytest tests/test_integration_local.py -q -s
```

The normal suite never contacts Anthropic. The optional
`ANTHROPIC_ITESTS=1 pytest tests/test_claude_client.py -q` run makes only the
explicit `count_tokens` cache-floor check and requires `ANTHROPIC_API_KEY`.
