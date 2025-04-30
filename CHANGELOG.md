# Changelog

Notable additions, fixes, or breaking changes to the Freeplay SDK.

## [0.3.17] - 2025-04-30

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

## [Before v0.3.17]

See https://docs.freeplay.ai/changelog