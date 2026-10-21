"""Run one analysis child with every network connection and name lookup refused.

Usage: python -I -X utf8 offline_stage.py <stage_id> <job_id>. Apart from the refusal it is the real child,
so a stage that passes here never needs the network.
"""

import socket
import sys

from fluentfolio.jobs import stage


def refuse(*args: object, **kwargs: object) -> None:
    raise OSError("network access from an analysis step")


socket.socket.connect = refuse  # type: ignore[method-assign]
socket.socket.connect_ex = refuse  # type: ignore[method-assign]
socket.getaddrinfo = refuse  # type: ignore[assignment]

if __name__ == "__main__":
    sys.exit(stage.main(sys.argv[1:]))
