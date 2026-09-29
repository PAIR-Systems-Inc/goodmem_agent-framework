# Copyright (c) Microsoft. All rights reserved.

"""Shared fixtures for the offline tests."""

from __future__ import annotations

import pytest

from ._fakes import FakeGoodMem


@pytest.fixture
def fake() -> FakeGoodMem:
    return FakeGoodMem()
