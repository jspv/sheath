"""Typed configuration for the tether."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def _default_sandbox_backend() -> str:
    """Shipped default: the container tier, i.e. real isolation.

    ``TETHER_SANDBOX_BACKEND`` overrides it. Setting it to ``local`` opts out of isolation
    entirely, so it is only for environments that have made that choice deliberately --
    CI and the test suite, which must run without a container runtime. An explicit
    ``SandboxConfig(backend=...)`` argument still wins over the variable.
    """
    return os.environ.get("TETHER_SANDBOX_BACKEND", "container")


@dataclass
class SandboxConfig:
    """How ``run_python`` code is executed.

    ``backend="local"`` runs code in a scrubbed-env subprocess with rlimits. It is **not a
    security boundary**: the code runs as the host user with the host user's file and network
    access. Use it only for development or fully trusted input. ``backend="container"`` runs
    code in a hardened OCI container (no network by default, read-only root filesystem, all
    capabilities dropped, ``no-new-privileges``, pids/memory/cpu limits, only the session root
    and the tether runtime mounted, no host environment).

    ``backend`` defaults to ``"container"``; ``TETHER_SANDBOX_BACKEND`` overrides that default
    (it selects the *tier*, not the container runtime). ``require_isolation=True`` is the
    stricter assertion on top: it refuses a local backend however it was selected -- an
    explicit ``backend="local"`` or an env-var-selected one -- at ``Session.create``, and a
    missing runtime or image raises, never falling back.
    ``build_on_demand`` controls whether a missing image or pip layer is built on first use;
    ``None`` means "build unless isolation is required" (a host requiring isolation should
    pre-build with ``tether-build-sandbox`` and gate startup on ``sandbox_preflight``).
    """

    timeout_s: float = 30.0
    max_memory_mb: int = 1024
    max_file_size_mb: int = 512        # enforced by the local tier only (no container equivalent)
    backend: Literal["local", "container"] = field(default_factory=_default_sandbox_backend)
    container_runtime: str | None = None   # None -> auto-detect podman, then docker
    network: bool = False                  # sandbox network off by default; opt-in to enable
    pip_packages: tuple[str, ...] = ()     # provisioned into a mounted layer (network only there)
    max_cpus: float = 2.0
    preinstalled: tuple[str, ...] = ("pandas", "pyarrow", "numpy", "httpx")
    require_isolation: bool = False        # refuse to run without a real boundary
    build_on_demand: bool | None = None    # None -> auto: build unless require_isolation
    build_timeout_s: float = 900.0         # bound on image build + pip layer provisioning
    max_output_bytes: int = 8192           # per stream (stdout, stderr): head + tail kept

    @property
    def effective_build_on_demand(self) -> bool:
        if self.build_on_demand is not None:
            return self.build_on_demand
        return not self.require_isolation


@dataclass
class FetchConfig:
    max_bytes: int = 10_000_000
    timeout_s: float = 30.0
    allowed_schemes: tuple[str, ...] = ("http", "https")
    # Hostnames or CIDRs exempted from the internal-address denylist. Empty by default:
    # internal data sources are a legitimate use case, but they must be named explicitly.
    allow_private_hosts: tuple[str, ...] = ()
    max_redirects: int = 5
    total_timeout_s: float = 120.0        # wall-clock bound on one fetch, all hops included
    max_bytes_ceiling: int = 100_000_000  # hard cap on a per-call max_bytes the model supplies


@dataclass
class SearchConfig:
    provider: str = "tavily"
    api_key: str | None = None
    max_results: int = 5
    timeout_s: float = 20.0


@dataclass
class DocumentConfig:
    # OCR is off by default: born-digital PDFs/Office files get tables and structure from
    # the layout/TableFormer models, so OCR only adds latency + model downloads. Turn it on
    # for scanned/image documents.
    ocr: bool = False


@dataclass
class TetherConfig:
    model: str = "gpt-5-mini"  # only used by the built-in OpenAI client; ignored when you inject a client
    spill_threshold_bytes: int = 8192          # lower edge: tool returns over this become handles
    max_spill_bytes: int = 100 * 1024 * 1024   # upper edge: a return over this is rejected, not stored
    max_context_window_tokens: int = 128_000
    max_output_tokens: int = 4096
    # Control-plane bounds (parent <-> sandbox channel). These live here rather than on
    # SandboxConfig because they bound the orchestration channel, not a backend's behavior.
    max_emit_bytes: int = 1024 * 1024          # emit payload cap, checked before parsing
    max_control_bytes: int = 8 * 1024 * 1024   # new-handles file read cap
    max_new_handles: int = 256                 # records adopted per run
    max_publish_bytes: int = 100 * 1024 * 1024  # largest file publish_file/publish() will deliver
    max_input_bytes: int = 100 * 1024 * 1024    # largest user upload add_input() will accept
    root_dir: Path | None = None  # None -> a session dir is created under ./.tether/sessions/
    cleanup: bool = False  # delete the root on async-context exit (throwaway runs)
    idle_ttl_s: float | None = None  # continuous-session idle TTL (None = never expire)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    documents: DocumentConfig = field(default_factory=DocumentConfig)
