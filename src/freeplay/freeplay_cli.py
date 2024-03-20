import json
import os
import sys
from pathlib import Path
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR

import click

from freeplay.errors import FreeplayClientError, FreeplayServerError
from freeplay import Freeplay
from freeplay.support import PromptTemplates, PromptTemplate, PromptTemplateEncoder


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--project-id", required=True, help="The Freeplay project ID.")
@click.option("--environment", required=True, help="The environment from which the prompts will be pulled.")
@click.option("--output-dir", required=True, help="The directory where the prompts will be saved.")
def download(project_id: str, environment: str, output_dir: str) -> None:
    if "FREEPLAY_API_KEY" not in os.environ:
        print("FREEPLAY_API_KEY is not set. It is required to run the freeplay command.", file=sys.stderr)
        exit(4)

    if "FREEPLAY_SUBDOMAIN" not in os.environ:
        print("FREEPLAY_SUBDOMAIN is not set. It is required to run the freeplay command.", file=sys.stderr)
        exit(4)

    FREEPLAY_API_KEY = os.environ["FREEPLAY_API_KEY"]
    freeplay_api_url = f'https://{os.environ["FREEPLAY_SUBDOMAIN"]}.freeplay.ai/api'

    if "FREEPLAY_API_URL" in os.environ:
        freeplay_api_url = f'{os.environ["FREEPLAY_API_URL"]}/api'
        click.echo("Using URL override for Freeplay specified in the FREEPLAY_API_URL environment variable")

    click.echo(
        f"Downloading prompts for project {project_id}, environment {environment}, "
        f"to directory {output_dir} from {freeplay_api_url}")

    fp_client = Freeplay(
        freeplay_api_key=FREEPLAY_API_KEY,
        api_base=freeplay_api_url
    )

    try:
        prompts: PromptTemplates = fp_client.prompts.get_all(project_id, environment=environment)
        click.echo(f"Found {len(prompts.prompt_templates)} prompt templates")

        for prompt in prompts.prompt_templates:
            __write_single_file(environment, output_dir, project_id, prompt)
    except FreeplayClientError as e:
        print(f"Error downloading templates: {e}.\nIs your project ID correct?", file=sys.stderr)
        exit(1)
    except FreeplayServerError as e:
        print(f"Error on Freeplay's servers downloading templates: {e}.\nTry again after a short wait.",
              file=sys.stderr)
        exit(2)
    except Exception as e:
        print(f"Error downloading templates: {e}", file=sys.stderr)
        exit(3)


def __write_single_file(
        environment: str,
        output_dir: str,
        project_id: str,
        prompt: PromptTemplate
) -> None:
    directory = __root_dir(environment, output_dir, project_id)
    basename = f'{prompt.prompt_template_name}'
    prompt_path = directory / f'{basename}.json'
    click.echo("Writing prompt file: %s" % prompt_path)

    # Make sure it's owner writable if it already exists
    if prompt_path.is_file():
        os.chmod(prompt_path, S_IWUSR | S_IREAD)

    with prompt_path.open(mode='w') as f:
        f.write(json.dumps(prompt, sort_keys=True, indent=4, cls=PromptTemplateEncoder))
        f.write('\n')

    # Make the file read-only to discourage local changes
    os.chmod(prompt_path, S_IREAD | S_IRGRP | S_IROTH)


def __root_dir(environment: str, output_dir: str, project_id: str) -> Path:
    directory = Path(output_dir) / "freeplay" / "prompts" / project_id / environment
    os.makedirs(directory, exist_ok=True)
    return directory
