# Changelog

Notable additions, fixes, or breaking changes to the Freeplay SDK.


## [0.3.20] - 2025-05-08

### Added

- Added support for files and audio in prompt templates.

## [0.3.19] - 2025-05-07

### Added

- Added support for images in prompt templates. Prompt templates created with media slots can be formatted using the Python SDK and sent as images to LLM providers using the media_inputs parameter:
```
self.freeplay_thin.prompts.get_formatted(
    project_id=self.project_id,
    template_name=template_name,
    environment=tag if tag else self.tag,
    variables=input_variables,
    media_inputs=media_inputs,
)
```
Future releases will include file inputs and audio inputs.


## [0.3.18] - 2025-04-30

### Added

- Enhanced agent support
    - `Session.create_trace` now accepts:
        - `agent_name`: used to name a "type" of trace and identify associated
          traces in the UI.
        - `custom_metadata`: used for logging of metadata from your execution environment.
          level like it is today.
    - `TraceInfo.record_output` now accepts:
        - `eval_results`: used to record evaluations
          similar to the output recorded on a completion.
- Added handling of prompt formatting for Perplexity models.

## [Before v0.3.18]

See https://docs.freeplay.ai/changelog