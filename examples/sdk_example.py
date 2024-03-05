import os
from typing import Generator

from freeplay import Freeplay
from freeplay.completions import CompletionChunk, ChatMessage
from freeplay.flavors import OpenAIChat, AzureOpenAIChat, AnthropicClaudeChat
from freeplay.provider_config import ProviderConfig, AnthropicConfig, OpenAIConfig, AzureConfig
from freeplay.record import no_op_recorder

openai_api_key = os.environ['OPENAI_API_KEY']
anthropic_key = os.environ['ANTHROPIC_API_KEY']
azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
freeplay_api_key = os.environ['EXAMPLES_FREEPLAY_API_KEY']
freeplay_project_id = os.environ['EXAMPLES_PROJECT_ID']
api_base = os.environ['EXAMPLES_API_URL']

azure_provider_config = AzureConfig(
    api_key=azure_openai_key,
    api_version='2023-05-15',
    engine="gpt-35-turbo"
)

provider_config = ProviderConfig(
    AnthropicConfig(anthropic_key),
    OpenAIConfig(openai_api_key),
    azure_provider_config
)


# ===========================================================================
# Chat ======================================================================
# ===========================================================================

# ===================================================
# Single Prompt Session with metadata
# - Creates an implicit session with a single prompt
# ===================================================
def run_openai_chat_single_prompt_session(freeplay_chat: Freeplay) -> None:
    chat_completion = freeplay_chat.get_completion(
        project_id=freeplay_project_id,
        template_name="my-chat-template",
        variables={"initial_user_input": "Why isn't my internet working?"},
        metadata={
            "customer_id": 123456,
            "gitSHA": "d5afe656acfedad35ef75eb55c8a1b853fcd1cd2",
        }
    )

    print(
        "Single-prompt completion (complete? %s): %s" % (chat_completion.is_complete, chat_completion.content.strip()))


# ===================================================
# Single Prompt Session - Does not record session
# - Creates an implicit session with a single prompt
# ===================================================
def run_openai_chat_single_prompt_session_not_record_session(freeplay_chat: Freeplay) -> None:
    chat_completion = freeplay_chat.get_completion(
        project_id=freeplay_project_id,
        template_name="my-chat-template",
        variables={"initial_user_input": "Why isn't my internet working?"}
    )

    print(
        "Single-prompt completion (complete? %s): %s" % (chat_completion.is_complete, chat_completion.content.strip()))


# ===================================================
# Multi-prompt Text session
# - Create a session then get multiple completions
# ===================================================
def run_openai_chat_multi_prompt_session(freeplay_chat: Freeplay) -> None:
    chat_session = freeplay_chat.create_session(freeplay_project_id)

    first_chat_completion = chat_session.get_completion(template_name="my-chat-prompt",
                                                        variables={"user_input": "Why isn't my internet working?"})
    print("\tFirst completion: %s" % first_chat_completion.content.strip())

    second_chat_completion = chat_session.get_completion(template_name="my-chat-prompt-2",
                                                         variables={"user_input": "Why isn't my internet working?"})
    print("\tSecond completion: %s" % second_chat_completion.content.strip())


# ===================================================
# Multi-prompt Text session
# - Create a session and get a completion, passing functions
# ===================================================
def run_openai_chat_multi_prompt_session_function_call(freeplay_chat: Freeplay) -> None:
    chat_session = freeplay_chat.create_session(freeplay_project_id)

    functions = [
        {
            "name": "get_album_tracklist",
            "description": "Given an album name and genre, return a list of songs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "album_name": {
                        "type": "string",
                        "description": "Name of album from which to retrieve tracklist."
                    },
                    "genre": {
                        "type": "string",
                        "description": "Album genre"
                    }
                }
            }
        }
    ]

    chat_completion = chat_session.get_completion(
        template_name="album_bot",
        variables={"pop_star": "Bruno Mars"},
        functions=functions,
        function_call={"name": "get_album_tracklist"}
    )

    print("First completion: ", chat_completion.content.strip())
    if chat_completion.openai_function_call:
        print("Function name: ", chat_completion.openai_function_call['name'])
        print("Function arguments: ", chat_completion.openai_function_call['arguments'])
    else:
        print("Error! No function returned!")
        exit(1)


# # ===================================================
# # Single Prompt Session - Streaming
# # ===================================================
def run_openai_chat_streaming(freeplay_chat: Freeplay) -> None:
    chat_stream = freeplay_chat.get_completion_stream(
        project_id=freeplay_project_id,
        template_name="my-chat-prompt",
        variables={"user_input": "Why isn't my internet working?"}
    )

    for chunk in chat_stream:
        print("Chunk: %s" % chunk.text.strip())


# # ===================================================
# # Single Prompt Session - Streaming - Function Call
# # ===================================================
def run_openai_chat_streaming_function_call(freeplay_chat: Freeplay) -> None:
    functions = [
        {
            "name": "get_album_tracklist",
            "description": "Given an album name and genre, return a list of songs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "album_name": {
                        "type": "string",
                        "description": "Name of album from which to retrieve tracklist."
                    },
                    "genre": {
                        "type": "string",
                        "description": "Album genre"
                    }
                }
            }
        }
    ]
    function_call = {"name": "get_album_tracklist"}

    chat_stream = freeplay_chat.get_completion_stream(
        project_id=freeplay_project_id,
        template_name="album_bot",
        variables={"pop_star": "Bruno Mars"},
        functions=functions,
        function_call=function_call)

    for chunk in chat_stream:
        print("Chunk: ", chunk.text.strip())
        print("Function name: ", chunk.openai_function_call.name if chunk.openai_function_call else '')
        print("Function arguments: ", chunk.openai_function_call.arguments if chunk.openai_function_call else '')


# ===================================================
# Continuous chat session
# - Create a session then get incremental completions as in a live chat UX.
# ===================================================
def run_openai_chat_ux_session(freeplay_chat: Freeplay) -> None:
    session, initial_completion = freeplay_chat.start_chat(project_id=freeplay_project_id,
                                                           template_name="my-chat-prompt",
                                                           variables={"user_input": "Why isn't my internet working?"},
                                                           tag='latest',
                                                           max_tokens=10)
    print('Initial chat completion is', initial_completion)

    second_chat_completion = session.continue_chat(new_messages=[{'role': 'user', 'content': 'Tell me more!'}],
                                                   max_tokens=10)
    print('second chat completion is', second_chat_completion.content)

    third_chat_completion = session.continue_chat(new_messages=[{'role': 'user', 'content': 'Now in spanish!'}],
                                                  max_tokens=10)
    print('third chat completion is', third_chat_completion.content)

    fourth_chat_completion = session.continue_chat(new_messages=[{'role': 'user', 'content': 'Now in JSON!'}],
                                                   max_tokens=50)
    print('fourth chat completion is', fourth_chat_completion.content)
    print("Final message chain", fourth_chat_completion.message_history)


# ===================================================
# Continuous chat session
# - Create a session then get incremental completions as in a live chat UX.
# ===================================================

def __message_from_streamed_response(chat_stream: Generator[CompletionChunk, None, None]) -> dict[str, str]:
    assistant_message = {'role': 'assistant', 'content': ''}
    for chunk in chat_stream:
        if chunk.text:
            print(chunk.text)
            assistant_message['content'] += chunk.text
    return assistant_message


def run_openai_chat_ux_session_streaming(freeplay_chat: Freeplay) -> None:
    session, initial_completion_stream = freeplay_chat.start_chat_stream(
        project_id=freeplay_project_id,
        template_name="my-chat-prompt",
        variables={"user_input": "Why isn't my internet working?"},
        tag='latest',
        max_tokens=10)

    __message_from_streamed_response(initial_completion_stream)

    first_user_message: ChatMessage = {'role': 'user', 'content': 'Tell me more!'}
    first_chat_stream = session.continue_chat_stream(new_messages=[first_user_message], max_tokens=10)
    __message_from_streamed_response(first_chat_stream)

    user_message_2: ChatMessage = {'role': 'user', 'content': 'Now in JSON!'}
    second_chat_stream = session.continue_chat_stream(new_messages=[user_message_2], max_tokens=10)
    __message_from_streamed_response(second_chat_stream)


# ===========================================================================
# Anthropic =================================================================
# ===========================================================================

# ===================================================
# Single Prompt Session
# - Creates an implicit session with a single prompt
# ===================================================
def run_anthropic_text_single_prompt_session(freeplay_anthropic_text: Freeplay) -> None:
    chat_completion = freeplay_anthropic_text.get_completion(
        project_id=freeplay_project_id,
        template_name="my-anthropic-prompt",
        variables={"user_input": "Why isn't my internet working?"}
    )

    print(
        "Single-prompt completion (complete? %s): %s" % (chat_completion.is_complete, chat_completion.content.strip()))


# # ===================================================
# # Multi-prompt Text session
# # - Create a session then get multiple completions
# # ===================================================
def run_anthropic_text_multi_prompt_session(freeplay_anthropic_text: Freeplay) -> None:
    chat_session = freeplay_anthropic_text.create_session(freeplay_project_id)
    first_chat_completion = chat_session.get_completion(template_name="my-anthropic-prompt",
                                                        variables={"user_input": "Why isn't my internet working?"},
                                                        max_tokens_to_sample=3)
    print("\tFirst completion: %s" % first_chat_completion.content.strip())

    second_chat_completion = chat_session.get_completion(template_name="my-anthropic-prompt-2",
                                                         variables={"user_input": "Why isn't my internet working?"})
    print("\tSecond completion: %s" % second_chat_completion.content.strip())


# ===================================================
# Single Prompt Session - Streaming - claude-instant-1 model
# ===================================================
def run_anthropic_text_stream(freeplay_anthropic_text: Freeplay) -> None:
    anthropic_stream = freeplay_anthropic_text.get_completion_stream(
        project_id=freeplay_project_id,
        template_name="my-anthropic-instant-prompt",
        variables={"user_input": "Why isn't my internet working?"},
    )

    for chunk in anthropic_stream:
        print("Chunk: %s" % chunk.text.strip())


def run_anthropic_chat(fp: Freeplay) -> None:
    session, stream = fp.start_chat_stream(
        project_id=freeplay_project_id,
        template_name="my-chat-prompt",
        variables={"user_input": "Why isn't my internet working?"},
        model='claude-2.1',
    )
    for chunk in stream:
        print("Chunk: %s" % chunk.text.strip())

    stream = fp.get_completion_stream(
        project_id=freeplay_project_id,
        template_name="my-chat-prompt",
        variables={"user_input": "Why isn't my internet working?"},
        flavor=AnthropicClaudeChat(),
        model='claude-2.1',
    )
    for chunk in stream:
        print("Chunk: %s" % chunk.text.strip())


# Run config ==========================================
_freeplay_chat = Freeplay(
    flavor=OpenAIChat(),
    provider_config=provider_config,
    freeplay_api_key=freeplay_api_key,
    api_base=api_base)

_freeplay_chat_no_record = Freeplay(
    flavor=OpenAIChat(),
    provider_config=provider_config,
    freeplay_api_key=freeplay_api_key,
    api_base=api_base,
    record_processor=no_op_recorder
)

_freeplay_anthropic = Freeplay(
    flavor=AnthropicClaudeChat(),
    provider_config=provider_config,
    freeplay_api_key=freeplay_api_key,
    api_base=api_base,
    max_tokens_to_sample=100)

anthropic_chat = Freeplay(
    flavor=AnthropicClaudeChat(),
    provider_config=provider_config,
    freeplay_api_key=freeplay_api_key,
    api_base=api_base,
    max_tokens_to_sample=100)

run_openai_chat_single_prompt_session_not_record_session(_freeplay_chat_no_record)

run_openai_chat_single_prompt_session(_freeplay_chat)
run_openai_chat_multi_prompt_session(_freeplay_chat)
run_openai_chat_streaming(_freeplay_chat)
run_openai_chat_ux_session(_freeplay_chat)
run_openai_chat_ux_session_streaming(_freeplay_chat)

run_openai_chat_multi_prompt_session_function_call(_freeplay_chat)
run_openai_chat_streaming_function_call(_freeplay_chat)

run_anthropic_text_single_prompt_session(_freeplay_anthropic)
run_anthropic_text_multi_prompt_session(_freeplay_anthropic)
run_anthropic_text_stream(_freeplay_anthropic)
run_anthropic_chat(anthropic_chat)
