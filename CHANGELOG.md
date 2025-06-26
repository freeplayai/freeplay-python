# Changelog

Notable additions, fixes, or breaking changes to the Freeplay SDK.

## 0.4.0 - 2025-06-26

### Breaking change
- `customer_feedback.update_customer_feedback()` now requires a project_id parameter.

## 0.3.25 - 2025-06-24

### Added
- New `download-all` CLI command that downloads all prompts across all projects within an account for bundling. Example:
    ```bash
    freeplay download-all --environment latest --output-dir ./prompts
    ```
    This command automatically downloads all of prompts from all projects tagged with the given [environment](https://docs.freeplay.ai/docs/managing-prompts#specifying-environments).

## 0.3.24 - 2025-05-29

### Added
- Create test run with dataset that targets agent. Example: 
    ```python
    test_run = fp_client.test_runs.create(
        project_id,
        "Dataset Name",
        include_outputs=True,
        name="Test run title",
        description='Some description',
        flavor_name=template_prompt.prompt_info.flavor_name
    )
    ```
- Use traces when creating test run. Example:
    ```python
    trace_info.record_output(
        project_id,
        completion.choices[0].message.content,
        {
            'f1-score': 0.48,
            'is_non_empty': True
        },
        test_run_info=test_run.get_test_run_info(test_case.id)
    )
    ```

### Updated
- Renamed `TestCase` dataclass to `CompletionTestCase` dataclass. The old `TestCase` is still exported as `TestCase` for backwards-compatibility, but is deprecated.
- Both `CompletionTestCase` and `TraceTestCase` now surface `custom_metadata` field if it was supplied when the dataset was built.


## [0.3.22] - 2025-05-22

### Fixed

- Allow passing provider specific messages in Gemini so history works.

## [0.3.22] - 2025-05-15

### Added

- Add support for Amazon Bedrock Converse flavor


## [0.3.21] - 2025-05-08

### Updated

- Updated "click" project dependency to support newer minor and patch versions.

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