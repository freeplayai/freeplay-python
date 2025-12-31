<h1 align="center">Freeplay Python SDK</h1>

<p align="center">
  <strong>The official Python SDK for <a href="https://freeplay.ai">Freeplay</a></strong><br/>
  The ops platform for enterprise AI engineering teams
</p>

<p align="center">
  <a href="https://pypi.org/project/freeplay/"><img src="https://img.shields.io/pypi/v/freeplay.svg" alt="version" /></a>
  <a href="https://github.com/freeplayai/freeplay-python/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License" /></a>
</p>

<p align="center">
  <a href="https://docs.freeplay.ai">Docs</a> •
  <a href="https://docs.freeplay.ai/getting-started/overview">Quick Start</a> •
  <a href="https://docs.freeplay.ai/freeplay-sdk/setup">SDK Setup</a> •
  <a href="https://docs.freeplay.ai/resources/api-reference">API Reference</a> •
  <a href="https://github.com/freeplayai/freeplay-python/blob/main/CHANGELOG.md">Changelog</a> •
  <a href="https://github.com/freeplayai/freeplay-python/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

---

## Overview

Freeplay is the only platform your team needs to manage the end-to-end AI application development lifecycle. It provides an integrated workflow for improving your AI agents and other generative AI products. Engineers, data scientists, product managers, designers, and subject matter experts can all review production logs, curate datasets, experiment with changes, create and run evaluations, and deploy updates.

Use this SDK to integrate with Freeplay's core capabilities:

**Observability**
- [**Sessions**](https://docs.freeplay.ai/freeplay-sdk/sessions) — group related interactions together, e.g. for multi-turn chat or complex agent interactions
- [**Traces**](https://docs.freeplay.ai/freeplay-sdk/traces) — track multi-step agent workflows within sessions
- [**Completions**](https://docs.freeplay.ai/freeplay-sdk/recording-completions) — record LLM interactions for observability and debugging
- [**Customer Feedback**](https://docs.freeplay.ai/freeplay-sdk/customer-feedback) — append user feedback and events to traces and completions
- [**Prompts**](https://docs.freeplay.ai/freeplay-sdk/prompts) — version, format, and fetch prompt templates across environments
- [**Test Runs**](https://docs.freeplay.ai/freeplay-sdk/test-runs) — execute evaluation runs against prompts and datasets

## Requirements

- Python 3.8 or higher
- A Freeplay account + API key

## Installation

```bash
pip install freeplay
```

## Quick Start

```python
import os
from freeplay import Freeplay, RecordPayload
from openai import OpenAI

fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
)
openai_client = OpenAI()

project_id = os.environ["FREEPLAY_PROJECT_ID"]

# Fetch a prompt from Freeplay
formatted_prompt = fp_client.prompts.get_formatted(
    project_id=project_id,
    template_name="my-prompt",
    environment="prod",
    variables={"user_input": "Hello, world!"},
)

# Call your LLM provider with formatted_prompt.llm_prompt
response = openai_client.chat.completions.create(
    model=formatted_prompt.prompt_info.model,
    messages=formatted_prompt.llm_prompt,
)

# Record the interaction for observability
fp_client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=formatted_prompt.all_messages({
            "role": "assistant",
            "content": response.choices[0].message.content,
        }),
    )
)
```

See the [SDK Setup guide](https://docs.freeplay.ai/freeplay-sdk/setup) for complete examples.

## Configuration

### Environment variables

```bash
export FREEPLAY_API_KEY="fp_..."
export FREEPLAY_PROJECT_ID="xy..."
# Optional: override if using a custom domain / private deployment
export FREEPLAY_API_BASE="https://app.freeplay.ai/api"
```

**API base URL**  
Default: `https://app.freeplay.ai/api`

Custom domain/private deployment: `https://<your-domain>/api`

## Versioning

This SDK follows Semantic Versioning (SemVer): **MAJOR.MINOR.PATCH**.

- **PATCH**: bug fixes
- **MINOR**: backward-compatible features
- **MAJOR**: breaking changes

Before upgrading major versions, review the changelog.

## Support

- **Docs**: https://docs.freeplay.ai
- **Issues**: https://github.com/freeplayai/freeplay-python/issues
- **Security**: security@freeplay.ai

## Contributing

See [CONTRIBUTING.md](https://github.com/freeplayai/freeplay-python/blob/main/CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](https://github.com/freeplayai/freeplay-python/blob/main/LICENSE).
