from enum import Enum, auto

class Status(Enum):
    """Solver states for the solving process."""
    NOT_STARTED = auto()
    SUCCESS = auto()
    FAILURE = auto()

    def is_terminal(self) -> bool:
        """Return True if the status is a terminal state (success or failure)."""
        return self in (Status.SUCCESS, Status.FAILURE)