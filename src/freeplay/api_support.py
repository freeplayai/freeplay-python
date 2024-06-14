import json
import logging
import typing as t
from typing import Dict

import dacite
import requests
from requests import Response

from freeplay.utils import build_request_header

T = t.TypeVar("T")

logger = logging.getLogger(__name__)


def force_decode(target_type: t.Type[T], data: bytes) -> T:
    parsed_json = json.loads(data)
    return dacite.from_dict(target_type, parsed_json)


def try_decode(target_type: t.Type[T], data: bytes) -> t.Optional[T]:
    try:
        return force_decode(target_type, data)
    except Exception as e:
        logger.error(f'There was an error decoding the json, {e}')
        return None


def post(target_type: t.Type[T], api_key: str, url: str, payload: t.Optional[Dict[str, str]] = None) -> T:
    response = requests.post(
        url=url,
        headers=build_request_header(api_key),
        json=payload
    )

    if response.status_code != 201:
        raise Exception(f'Unexpected status code for POST at {url}, got {response.status_code}')

    maybe_object = try_decode(target_type, response.content)
    if maybe_object is None:
        raise Exception(f'Failed to parse response from POST at {url}')

    return maybe_object


def put_raw(api_key: str, url: str, payload: t.Optional[Dict[str, t.Any]] = None) -> Response:
    return requests.put(
        url=url,
        headers=build_request_header(api_key),
        json=payload
    )


def post_raw(api_key: str, url: str, payload: t.Optional[Dict[str, t.Any]] = None) -> Response:
    return requests.post(
        url=url,
        headers=build_request_header(api_key),
        json=payload
    )


def delete_raw(api_key: str, url: str) -> Response:
    return requests.delete(
        url=url,
        headers=build_request_header(api_key),
    )


def get(target_type: t.Type[T], api_key: str, url: str) -> T:
    response = requests.get(
        url=url,
        headers=build_request_header(api_key),
    )

    if response.status_code != 200:
        raise Exception(f'Unexpected status code for GET at {url}, got {response.status_code}')

    maybe_object = try_decode(target_type, response.content)
    if maybe_object is None:
        raise Exception(f'Failed to parse response from GET at {url}')

    return maybe_object


def get_raw(
        api_key: str,
        url: str,
        params: t.Optional[Dict[str, str]] = None
) -> Response:
    return requests.get(
        url=url,
        headers=build_request_header(api_key),
        params=params
    )
