"""Allow ``python -m lithos_repair`` to dispatch to :mod:`lithos_repair.cli`."""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":                           # pragma: no cover
    sys.exit(main(sys.argv[1:]))
