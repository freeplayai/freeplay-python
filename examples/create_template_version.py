import os

from customer_utils import get_freeplay_thin_client
from freeplay.support import TemplateChatMessage, ToolSchema

fp_client = get_freeplay_thin_client()

project_id = os.environ["FREEPLAY_PROJECT_ID"]
template_name = os.environ["PROMPT_TEMPLATE_NAME"]

tool_schemas = [
    ToolSchema(
        name="get_album_tracklist",
        description="Given an album name and genre, return a list of songs.",
        parameters={
            "type": "object",
            "properties": {
                "album_name": {
                    "type": "string",
                    "description": "Name of the album from which to retrieve tracklist.",
                },
                "genre": {"type": "string", "description": "Album genre"},
            },
        },
    )
]

created_version = fp_client.prompts.create_version(
    project_id=project_id,
    template_name=template_name,
    template_messages=[
        TemplateChatMessage(
            "user", "Answer this question as pythonic as you can please {{question}}"
        )
    ],
    provider="anthropic",
    model="claude-4-sonnet-20250514",
    tool_schema=tool_schemas,
)
print(f"Created version: {created_version}")

fp_client.prompts.update_version_environments(
    project_id=project_id,
    template_id=created_version.prompt_template_id,
    template_version_id=created_version.prompt_template_version_id,
    environments=["dev"],
)
