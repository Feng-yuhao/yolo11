#!/usr/bin/env python3
"""C13 paired control: fine-tune only E10a P2/P3 classification towers for 30 epochs."""
from e13_pair_common import run


if __name__ == "__main__":
    run("control", __file__)
