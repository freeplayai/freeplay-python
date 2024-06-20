from requests import Response
import warnings


class FreeplayError(Exception):
    pass


class FreeplayConfigurationError(FreeplayError):
    pass


class FreeplayClientError(FreeplayError):
    pass


class FreeplayServerError(FreeplayError):
    pass


class LLMClientError(FreeplayError):
    pass


class LLMServerError(FreeplayError):
    pass


def freeplay_response_error(message: str, response: Response) -> FreeplayError:
    full_message = f'{message} [{response.status_code}]'

    if response.status_code in range(400, 500):
        return FreeplayClientError(full_message)
    else:
        return FreeplayServerError(full_message)


class FreeplayClientWarning(UserWarning):
    pass


def log_freeplay_client_warning(message: str) -> None:
    warnings.warn(message, FreeplayClientWarning)
