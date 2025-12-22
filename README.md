# Freeplay Python SDK 

The official Python SDK for easily accessing the Freeplay API.

## Installation

```
pip install freeplay
```

## Compatibility

- Python 3.8+

## Usage

```python
# Import the SDK
from freeplay import Freeplay

# Initialize the client
fp_client = Freeplay(
    provider_config=ProviderConfig(openai=OpenAIConfig(OPENAI_API_KEY)),
    freeplay_api_key=FREEPLAY_API_KEY,
    api_base=f'https://{FREEPLAY_CUSTOMER_NAME}.freeplay.ai/api')

# Completion Request
completion = fp_client.get_completion(project_id=FREEPLAY_PROJECT_ID,
                                      template_name="template",
                                      variables={"input_variable_name": "input_variable_value"})
```

See the [Freeplay Docs](https://docs.freeplay.ai) for more usage examples and the API reference.

## Updating Metadata

You can update session and trace metadata at any point after creation. This is useful when you need to associate IDs or information that's generated after a conversation ends:

```python
# Update session metadata
fp_client.metadata.update_session(
    project_id=project_id,
    session_id=session_id,
    metadata={
        "customer_id": "cust_123",
        "conversation_rating": 5,
        "support_tier": "premium"
    }
)

# Update trace metadata
fp_client.metadata.update_trace(
    project_id=project_id,
    session_id=session_id,
    trace_id=trace_id,
    metadata={
        "agent_name": "customer_support_bot",
        "resolved": True,
        "resolution_time_ms": 1234
    }
)
```

**Merge semantics**: New keys overwrite existing keys, while unmentioned keys are preserved.

See `examples/update_metadata.py` for a complete example.

## License

This SDK is released under the [MIT License](LICENSE).
