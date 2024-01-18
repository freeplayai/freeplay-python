import uuid

from freeplay.thin.model import Session


class Sessions:
    # noinspection PyMethodMayBeStatic
    def create(self) -> Session:
        return Session(session_id=str(uuid.uuid4()))
