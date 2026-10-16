"""Data-integration agent harness."""

from .api import Tether, Result, ToolFactory, solve, tool_factory
from .config import FetchConfig, TetherConfig, SandboxConfig
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

__version__ = "0.1.1"

__all__ = [
    "Tether", "Result", "solve", "ToolFactory", "tool_factory",
    "FetchConfig", "TetherConfig", "SandboxConfig",
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
