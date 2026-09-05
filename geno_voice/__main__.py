"""Allow ``python -m geno_voice`` to behave like ``geno-voice``."""

import sys

from geno_voice.cli import main


if __name__ == "__main__":
    sys.exit(main())
