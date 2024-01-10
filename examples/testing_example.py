import os

from freeplay import Freeplay
from freeplay.provider_config import ProviderConfig, OpenAIConfig

openai_api_key = os.environ['OPENAI_API_KEY']
project_id = os.environ['EXAMPLES_PROJECT_ID']
freeplay_api_key = os.environ['EXAMPLES_FREEPLAY_API_KEY']
playlist_name = 'core-tests'

freeplay = Freeplay(
    provider_config=ProviderConfig(openai=OpenAIConfig(openai_api_key)),
    freeplay_api_key=freeplay_api_key,
    api_base=os.environ['EXAMPLES_API_URL']
)

test_run = freeplay.create_test_run(project_id, testlist=playlist_name)

for inputs in test_run.get_inputs():
    freeplay_session = test_run.create_session(project_id)

    completion = freeplay_session.get_completion(
        'my-chat-template', variables=inputs
    )
    print(completion.content)
