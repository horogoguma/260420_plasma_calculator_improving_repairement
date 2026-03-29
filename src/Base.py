"""PySpice/ngspice runtime setup with cross-platform defaults."""

from __future__ import annotations

import ctypes.util
import os
import platform
from pathlib import Path

from PySpice.Spice.NgSpice.Shared import NgSpiceShared

WINDOWS_DEFAULT_NG_ROOT = Path(r"C:\ngspice-31_64\Spice64")
# Backward compatibility for legacy imports.
NG_ROOT = str(WINDOWS_DEFAULT_NG_ROOT)
_ENV_INITIALIZED = False


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    for env_key in ("NGSPICE_ROOT", "NGSPICE_HOME"):
        value = os.environ.get(env_key)
        if value:
            roots.append(Path(value))

    system_name = platform.system().lower()
    if system_name == "windows":
        roots.append(WINDOWS_DEFAULT_NG_ROOT)
    elif system_name == "linux":
        roots.extend((Path("/usr"), Path("/usr/local"), Path("/opt")))
    elif system_name == "darwin":
        roots.extend((Path("/opt/homebrew"), Path("/usr/local"), Path("/opt/local")))
    return roots


def _library_candidates_from_root(root: Path) -> list[Path]:
    system_name = platform.system().lower()
    if system_name == "windows":
        return [root / "bin" / "ngspice.dll"]
    if system_name == "darwin":
        return [root / "lib" / "libngspice.dylib"]
    return [
        root / "lib" / "libngspice.so",
        root / "lib64" / "libngspice.so",
        root / "lib" / "ngspice" / "libngspice.so",
    ]


def _resolve_library_path() -> str | None:
    env_library = os.environ.get("NGSPICE_LIBRARY_PATH")
    if env_library:
        return env_library

    for root in _candidate_roots():
        for candidate in _library_candidates_from_root(root):
            if candidate.exists():
                return str(candidate)

    found = ctypes.util.find_library("ngspice")
    if found:
        # On Linux this is often "libngspice.so.0", which dlopen accepts.
        return found
    return None


def _resolve_spice_lib_dir() -> str | None:
    env_spice_lib_dir = os.environ.get("SPICE_LIB_DIR")
    if env_spice_lib_dir:
        return env_spice_lib_dir

    for root in _candidate_roots():
        candidate = root / "share" / "ngspice"
        if candidate.exists():
            return str(candidate)
    return None


def initialize_pyspice() -> None:
    """Configure process environment variables used by PySpice."""
    global _ENV_INITIALIZED
    if _ENV_INITIALIZED:
        return

    library_path = _resolve_library_path()
    if library_path:
        os.environ["NGSPICE_LIBRARY_PATH"] = library_path
        NgSpiceShared.LIBRARY_PATH = library_path

        # Prepend absolute library directory for dynamic linker lookup.
        parent = Path(library_path).parent
        if str(parent) not in ("", "."):
            os.environ["PATH"] = str(parent) + os.pathsep + os.environ.get("PATH", "")
            if platform.system().lower() == "windows" and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(parent))

    spice_lib_dir = _resolve_spice_lib_dir()
    if spice_lib_dir:
        os.environ["SPICE_LIB_DIR"] = spice_lib_dir

    _ENV_INITIALIZED = True


# Default setup on import for existing call sites.
initialize_pyspice()
