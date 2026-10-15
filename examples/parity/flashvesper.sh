#!/bin/bash
# VESPER=examples/parity/flashvesper.sh: FlashVESPER's command with `vesper orig`'s defaults. It has no
# `orig`, and its defaults are 3 A voxels and duplicate removal; an example's own -s or -nodup comes later and wins.
[ "$1" = orig ] && shift
exec flashvesper -s 7 -dup "$@"
