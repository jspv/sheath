"""Data-integration agent harness."""

from .api import Sheath, Result, ToolFactory, solve, tool_factory
from .config import FetchConfig, SheathConfig, SandboxConfig
from .handles import Handle, HandleStore
from .paths import PathEscapesRootError, safe_path
from .publish import OnPublish, PublishedFile, PublishError
from .container_runtime import (PreflightReport, SandboxImageError, SandboxImageMissing,
                                sandbox_preflight)
from .sandbox import (ExecResult, LocalSubprocessSandbox, NoSandboxIsolationWarning,
                      SandboxExecutor, SandboxQuarantinedError,
                      SandboxRuntimeUnavailable)
from .sandbox_container import ContainerSandbox
from .session import Session
from .conversation import Conversation
from .manager import SessionManager
from .status import StatusBus, StatusEvent, report_progress
from .tools.registry import build_tools

__version__ = "0.2.0"

__all__ = [
    "Sheath", "Result", "solve", "ToolFactory", "tool_factory",
    "FetchConfig", "SheathConfig", "SandboxConfig",
    "Handle", "HandleStore",
    "PathEscapesRootError", "safe_path",
    "OnPublish", "PublishedFile", "PublishError",
    "ExecResult", "LocalSubprocessSandbox", "SandboxExecutor", "SandboxQuarantinedError",
    "SandboxRuntimeUnavailable",
    "NoSandboxIsolationWarning",
    "PreflightReport", "SandboxImageError", "SandboxImageMissing", "sandbox_preflight",
    "ContainerSandbox",
    "Session",
    "Conversation", "SessionManager",
    "StatusBus", "StatusEvent", "report_progress",
    "build_tools",
]
