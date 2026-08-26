"""Installed ``geno-voice`` / ``gv`` console entrypoint."""

from __future__ import annotations


def main(argv=None) -> int:
    # Keep parser construction light; audio/model modules are imported only by
    # the selected command handler.
    from examples.gv import main as gv_main

    return gv_main(argv, prog="geno-voice")
