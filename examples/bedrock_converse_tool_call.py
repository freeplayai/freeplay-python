import copy
import json
import os
import time
from typing import Any, Dict, List, Union
import boto3  # type: ignore

from freeplay import CallInfo, Freeplay, RecordPayload


def add_numbers(numbers: List[int]) -> int:
    return sum(numbers)


def multiple_numbers(numbers: List[int]) -> int:
    result = 1
    for number in numbers:
        result *= number
    return result


def subtract_two_numbers(a: int, b: int) -> int:
    return a - b


def divide_two_numbers(a: int, b: int) -> float:
    return a / b


def execute_function(func_name: str, args: Dict[str, Any]) -> Union[int, float]:
    if func_name == "add_numbers":
        return add_numbers(args["numbers"])
    elif func_name == "multiple_numbers":
        return multiple_numbers(args["numbers"])
    elif func_name == "subtract_two_numbers":
        return subtract_two_numbers(args["a"], args["b"])
    elif func_name == "divide_two_numbers":
        return divide_two_numbers(args["a"], args["b"])
    else:
        raise Exception("Function not found")


toolsSpec = [
    {
        "toolSpec": {
            "name": "add_numbers",
            "description": "Add a list of numbers",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of numbers to add",
                        }
                    },
                    "required": ["numbers"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "multiple_numbers",
            "description": "Multiply a list of numbers",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "numbers": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of numbers to multiply",
                        }
                    },
                    "required": ["numbers"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "subtract_two_numbers",
            "description": "Subtract two numbers",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First number"},
                        "b": {"type": "integer", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "divide_two_numbers",
            "description": "Divide two numbers",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First number"},
                        "b": {"type": "integer", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                }
            },
        }
    },
]


fp_client = Freeplay(
    api_base=f"{os.environ.get('FREEPLAY_API_URL')}/api",
    freeplay_api_key=os.environ.get("FREEPLAY_API_KEY") or "",
)
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID") or ""
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or ""
if not aws_access_key_id or not aws_secret_access_key:
    raise Exception("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")
converse_client = boto3.client(  # type: ignore
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)
project_id: str = os.environ["FREEPLAY_PROJECT_ID"]

equation = "2x + 5 = 10"
prompt_vars = {"equation": equation}

try:
    formatted_prompt = fp_client.prompts.get_formatted(
        project_id=project_id,
        template_name="nova_tool_call",
        environment="latest",
        variables=prompt_vars,
    )
except Exception as e:
    print(f"Error details: {e}")
    print(f"Project ID: {project_id}")
    print("Template: nova_tool_call")
    print("Environment: latest")
    raise

session = fp_client.sessions.create()
trace = session.create_trace(input=equation)

print(f"Using model: {formatted_prompt.prompt_info.model}")
print(f"Template: {formatted_prompt.prompt_info.template_name}")

# Initialize history with the formatted prompt messages
history = copy.deepcopy(formatted_prompt.llm_prompt)

finish_reason = None
while finish_reason not in ["end_turn", "stop"]:
    s = time.time()
    print(f"Model: {formatted_prompt.prompt_info.model}")

    # Make the Bedrock Converse call
    response = converse_client.converse(  # type: ignore
        modelId=formatted_prompt.prompt_info.model,
        messages=history,
        system=[{"text": formatted_prompt.system_content or ""}],
        inferenceConfig=formatted_prompt.prompt_info.model_parameters,
        toolConfig={"tools": toolsSpec},
    )
    e = time.time()
    print(f"Response: {json.dumps(response, indent=2, default=str)}")

    output_message = response["output"]["message"]  # type: ignore
    finish_reason = response["stopReason"]  # type: ignore

    print(f"Stop reason: {finish_reason}")

    # Check for tool calls
    if finish_reason == "tool_use":
        # Find the toolUse in content (may not be first item due to thinking text)
        tool_use = None
        for content_item in output_message["content"]:  # type: ignore
            if "toolUse" in content_item:
                tool_use = content_item["toolUse"]  # type: ignore
                break

        if not tool_use:
            raise Exception("No toolUse found in response")
        tool_name = tool_use["name"]  # type: ignore
        tool_input = tool_use["input"]  # type: ignore
        tool_id = tool_use["toolUseId"]  # type: ignore

        print(
            f"Executing function {tool_name} with args {json.dumps(tool_input, indent=2)}"
        )
        result: Union[int, float] = execute_function(tool_name, tool_input)  # type: ignore
        print(f"Result: {result}")
        print("\n")

        # Add the full assistant response to history (includes thinking + toolUse)
        print("=== Adding assistant message to history ===")
        print(f"Assistant message: {json.dumps(output_message, indent=2, default=str)}")
        history.append(output_message)

        # Add the tool response to history
        tool_result_message: Dict[str, Any] = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": tool_id,
                        "content": [{"text": str(result)}],
                    }
                }
            ],
        }
        print("\n=== Adding tool result to history ===")
        print(
            f"Tool result message: {json.dumps(tool_result_message, indent=2, default=str)}"
        )
        history.append(tool_result_message)

        # Record the tool call to freeplay
        print("\n=== Recording to Freeplay ===")
        print(f"History length: {len(history)}")
        print("Full history:")
        print(json.dumps(history, indent=2, default=str))
        print("\nTool schema being sent:")
        print(json.dumps(toolsSpec, indent=2, default=str))

        payload = RecordPayload(
            project_id=project_id,
            all_messages=history,
            inputs=prompt_vars,
            session_info=session.session_info,
            trace_info=trace,
            prompt_version_info=formatted_prompt.prompt_info,
            call_info=CallInfo.from_prompt_info(
                formatted_prompt.prompt_info, start_time=s, end_time=e
            ),
        )

        print("\n=== Payload details ===")
        print(f"Project ID: {project_id}")
        print(f"Session ID: {session.session_info.session_id}")
        print("Is complete: False")
        print(f"Function call: {tool_name}({json.dumps(tool_input)})")

        try:
            fp_client.recordings.create(payload)
            print("\n✓ Successfully recorded to Freeplay")
        except Exception as e:
            print(f"\n✗ Error recording to Freeplay: {e}")
            raise
    else:
        # Final response
        content = output_message["content"][0]["text"]  # type: ignore
        print("=== Solution ===")
        print(content)  # type: ignore
        print("\n")

        # Add the final response to history
        print("=== Adding final response to history ===")
        print(f"Final message: {json.dumps(output_message, indent=2, default=str)}")
        history.append(output_message)

        # Record the final response to freeplay
        print("\n=== Recording final response to Freeplay ===")
        print(f"History length: {len(history)}")
        print("Full history:")
        print(json.dumps(history, indent=2, default=str))

        payload = RecordPayload(
            project_id=project_id,
            all_messages=history,
            inputs=prompt_vars,
            session_info=session.session_info,
            trace_info=trace,
            prompt_version_info=formatted_prompt.prompt_info,
            call_info=CallInfo.from_prompt_info(
                formatted_prompt.prompt_info, start_time=s, end_time=e
            ),
        )

        print("\n=== Payload details ===")
        print(f"Project ID: {project_id}")
        print(f"Session ID: {session.session_info.session_id}")
        print("Is complete: True")

        try:
            fp_client.recordings.create(payload)
            print("\n✓ Successfully recorded to Freeplay")
        except Exception as e:
            print(f"\n✗ Error recording to Freeplay: {e}")
            raise

        trace.record_output(project_id=project_id, output=content)  # type: ignore
