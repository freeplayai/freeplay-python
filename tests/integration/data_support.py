import base64
from pathlib import Path


def encode_test_data(filename: str) -> str:
    path = Path(__file__).parent / "test_data" / filename
    with open(path, 'rb') as file:
        file_data = file.read()
        return base64.b64encode(file_data).decode('utf-8')
