# Changelog

Notable additions, fixes, or breaking changes to the Freeplay SDK.

## [0.5.7]

### Added

- **GenAI Tool Schema Support**: Added `GenaiFunction` and `GenaiTool` dataclasses for Google GenAI/Vertex AI tool schema format.

  ```python
  from freeplay import GenaiFunction, GenaiTool

  # Create function declaration
  weather_function = GenaiFunction(
      name="get_weather",
      description="Get the current weather for a location",
      parameters={
          "type": "object",
          "properties": {
              "location": {"type": "string", "description": "City name"},
              "units": {
                  "type": "string",
                  "enum": ["celsius", "fahrenheit"],
                  "description": "Temperature units"
              }
          },
          "required": ["location"]
      }
  )

  # Create tool schema (single tool with multiple functions)
  tool_schema = [GenaiTool(function_declarations=[weather_function])]

  # Use in recordings
  client.recordings.create(
      RecordPayload(
          project_id=project_id,
          all_messages=[...],
          tool_schema=tool_schema,
          call_info=CallInfo(provider="vertex", model="gemini-2.0-flash")
      )
  )
  ```

  **Key Features:**
  - Supports GenAI's unique format: single tool with multiple function declarations
  - Full type safety with Python dataclasses
  - Backward compatible: existing tool schema formats continue to work
  - Comprehensive test coverage

  **Notes:**
  - GenAI uses a different structure than OpenAI/Anthropic (multiple functions per tool object)
  - Same format works for both GenAI API and Vertex AI
  - Backend automatically normalizes all tool schema formats

- Interactive REPL (`make repl`) for development and testing with:
  - Pre-loaded imports (Freeplay, GenaiFunction, GenaiTool, etc.)
  - Environment variables automatically loaded from `.env` file
  - SSL verification disabled for local development
  - Pre-initialized `client` variable ready to use

## [0.5.6]

### Fixed

- Fixed broken links in README when viewed on PyPI (CHANGELOG, CONTRIBUTING, LICENSE now use absolute GitHub URLs)

## [0.5.5]

### Changed

- License changed from MIT to Apache-2.0 

### Added

- New `Metadata` resource for updating session and trace metadata after creation:

  ```python
  # Update session metadata
  fp_client.metadata.update_session(
      project_id=project_id,
      session_id=session_id,
      metadata={"customer_id": "cust_123", "rating": 5}
  )

  # Update trace metadata
  fp_client.metadata.update_trace(
      project_id=project_id,
      session_id=session_id,
      trace_id=trace_id,
      metadata={"resolved": True, "resolution_time_ms": 1234}
  )
  ```

  This addresses the use case where IDs or metadata are generated at the end of a conversation and need to be associated with existing sessions/traces without logging additional completions. Metadata updates use merge semantics - new keys overwrite existing keys while preserving unmentioned keys.

## [0.5.4] - 2025-11-07

- New examples around multimodal images as output
- Get test cases and output_message which may be more than just text content. 'output' is now deprecated.
- Add explicit tool span logging

## [0.5.3] - 2025-10-17

- Remove image and file restriction for Bedrock Converse.

## [0.5.2] - 2025-10-07

### Added

- New `parent_id` parameter in `RecordPayload` to replace the deprecated `trace_info` parameter. This UUID field enables direct parent-child trace/completions relationships:

  ```python
  # Before (deprecated):
  RecordPayload(
      project_id=project_id,
      all_messages=messages,
      trace_info=trace_info
  )

  # After:
  RecordPayload(
      project_id=project_id,
      all_messages=messages,
      parent_id=parent_id  # UUID of parent trace or completion
  )
  ```

- `parent_id` parameter support in `Session.create_trace()`:
  ```python
  parent_trace = session.create_trace(input="Parent question", agent_name="parent_agent")
  child_trace = session.create_trace(
      input="Child question",
      agent_name="child_agent",
      parent_id=uuid.UUID(parent_trace.trace_id) # Or it can be an ID of a completion
  )
  ```
- `parent_id` parameter in `Session.restore_trace()` method

### Change

- `RecordPayload.trace_info` parameter is deprecated and will be removed in v0.6.0. Use `parent_id` instead for trace hierarchy management.

## [0.5.0] - 2025-08-28

### Breaking changes

- `RecordPayload` now requires `project_id` as the first parameter. All code creating `RecordPayload` instances must be
  updated to include this field.
- `PromptInfo` no longer contains a `project_id` field. The project ID must now be accessed from the project context
  instead.
- `RecordPayload.prompt_info` field has been renamed to `RecordPayload.prompt_version_info` and now accepts `PromptVersionInfo` objects. Existing `PromptInfo` objects can still be passed, but the field name must be updated:

  ```python
  # Before:
  RecordPayload(
      project_id=project_id,
      all_messages=messages,
      prompt_info=formatted_prompt.prompt_info
  )

  # After:
  RecordPayload(
      project_id=project_id,
      all_messages=messages,
      prompt_version_info=formatted_prompt.prompt_info
  )
  ```

### Added

- New `PromptVersionInfo` class that provides lightweight prompt version information with only `prompt_template_version_id` and optional `environment` fields. `PromptInfo` now inherits from this class.
- Support for Vertex AI tool calling. Example:

  ```python
  from vertexai.generative_models import GenerativeModel

  # Get formatted prompt with tool schema
  formatted_prompt = fp_client.prompts.get(
      project_id=project_id,
      template_name='my-prompt',
      environment='latest'
  ).bind(input_variables).format()

  # Tool schema automatically converted to Vertex AI format
  model = GenerativeModel(
      model_name=formatted_prompt.prompt_info.model,
      tools=formatted_prompt.tool_schema  # Returns list[Tool] for Vertex AI
  )
  ```

- Add new optional field `target_evaluation_ids` to `TestRuns.create()` to control which evaluations run as part of a
  test.
- Test cases created via `create` or `create_many` may now specify `media_inputs` to programmatically create test cases
  with images, audio, and other files.

### Changed

- In `RecordPayload`, the following fields are now optional:
  - `inputs` (Optional)
  - `prompt_version_info` (Optional, renamed from `prompt_info`)
  - `call_info` (Optional)
- `session_info` in `RecordPayload` now has a default value and will be automatically generated if not provided.

## [0.4.1] - 2025-06-30

- Create a test run from the SDK with test cases with media in them.

## 0.4.0 - 2025-06-26

### Breaking change

- `customer_feedback.update_customer_feedback()` now requires a project_id parameter.

## 0.3.25 - 2025-06-24

### Added

- New `download-all` CLI command that downloads all prompts across all projects within an account for bundling. Example:
  ```bash
  freeplay download-all --environment latest --output-dir ./prompts
  ```
  This command automatically downloads all of prompts from all projects tagged with the
  given [environment](https://docs.freeplay.ai/docs/managing-prompts#specifying-environments).

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

- Renamed `TestCase` dataclass to `CompletionTestCase` dataclass. The old `TestCase` is still exported as `TestCase` for
  backwards-compatibility, but is deprecated.
- Both `CompletionTestCase` and `TraceTestCase` now surface `custom_metadata` field if it was supplied when the dataset
  was built.

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

- Added support for images in prompt templates. Prompt templates created with media slots can be formatted using the
  Python SDK and sent as images to LLM providers using the media_inputs parameter:

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
