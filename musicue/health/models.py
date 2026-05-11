from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel


class ComponentState(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    MISSING = "missing"
    ERROR = "error"


class ComponentStatus(BaseModel):
    name: str
    state: ComponentState
    required: bool
    version: str | None = None
    detail: str | None = None
    cache_path: str | None = None
    remediation: str | None = None


class ReadinessReport(BaseModel):
    components: list[ComponentStatus]
    overall: Literal["green", "amber", "red"]
    checked_at: datetime
