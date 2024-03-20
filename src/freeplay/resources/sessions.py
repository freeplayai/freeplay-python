import uuid
from dataclasses import dataclass
from typing import Optional, Dict, Union

CustomMetadata = Optional[Dict[str, Union[str, int, float]]]


@dataclass
class SessionInfo:
    session_id: str
    custom_metadata: CustomMetadata


@dataclass
class Session:
    session_id: str
    custom_metadata: CustomMetadata

    def __init__(self, session_id: str, custom_metadata: CustomMetadata):
        self.session_id = session_id
        self.custom_metadata = custom_metadata
        self._session_info = SessionInfo(self.session_id, self.custom_metadata)

    @property
    def session_info(self) -> SessionInfo:
        return self._session_info


class Sessions:
    # noinspection PyMethodMayBeStatic
    def create(self, custom_metadata: CustomMetadata = None) -> Session:
        return Session(session_id=str(uuid.uuid4()), custom_metadata=custom_metadata)
