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


## License

This SDK is released under the [MIT License](LICENSE).
