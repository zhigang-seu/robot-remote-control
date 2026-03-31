#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path


def get_sdk_root() -> Path:
    sdk_root = os.environ.get('BOOSTER_SDK_ROOT', '').strip()
    if not sdk_root:
        raise RuntimeError(
            'BOOSTER_SDK_ROOT is not set. Example: export BOOSTER_SDK_ROOT=/home/master/Workspace/test_lxk/booster_robotics_sdk-main'
        )
    p = Path(sdk_root).expanduser().resolve()
    if not p.exists():
        raise RuntimeError(f'BOOSTER_SDK_ROOT does not exist: {p}')
    return p


def add_sdk_paths() -> Path:
    root = get_sdk_root()
    low_level = root / 'example' / 'low_level'
    for p in [root, low_level]:
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    return root
