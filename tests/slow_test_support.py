import os
from typing import Callable


def slow(test_function: Callable[..., None]) -> Callable[..., None]:
    def wrapper(self, *args, **kwargs) -> None: # type: ignore
        if os.environ.get("RUN_SLOW_TESTS") == "true":
            test_function(self, *args, **kwargs)
        else:
            self.skipTest("skipping slow test")

    return wrapper
