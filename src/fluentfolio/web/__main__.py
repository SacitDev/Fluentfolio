"""The process that `fluentfolio serve` starts: python -m fluentfolio.web [--port N] [--no-browser]."""

import sys

from fluentfolio.web.server import main

sys.exit(main())
