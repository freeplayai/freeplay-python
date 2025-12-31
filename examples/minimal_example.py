import os
import time

from openai import OpenAI
from freeplay import Freeplay, RecordPayload

fp_client = Freeplay(
    freeplay_api_key=os.environ["FREEPLAY_API_KEY"],
    api_base=f"{os.environ['FREEPLAY_API_URL']}/api",
)
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
project_id = os.environ["FREEPLAY_PROJECT_ID"]
name = "John"


start = time.time()
all_messages = [
    {
        "role": "system",
        "content": "You just say good job when someone tells their name. Like 'Good job, <name>!'",
    },
    {"role": "user", "content": f"My name is {name}."},
]
completion = openai_client.chat.completions.create(
    messages=all_messages, model="gpt-3.5-turbo"
)
end = time.time()


fp_client.recordings.create(
    RecordPayload(
        project_id=project_id,
        all_messages=all_messages + [completion.choices[0].message.model_dump()],
        # inputs={"name": name},
    )
)
