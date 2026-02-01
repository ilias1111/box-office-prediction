---
name: uv-package-manager
description: Instructions for using uv for fast Python package management
---

# uv Package Management

`uv` is a blazing fast Python package installer and resolver, designed as a drop-in replacement for `pip` and `pip-tools`.

## Usage Rules

1.  **Prefer uv**: Whenever you need to install packages, create virtual environments, or resolve dependencies, use `uv` instead of standard `pip` or `venv` commands.
2.  **Virtual Environments**:
    *   Create: `uv venv` (creates `.venv` by default)
    *   Activate: `source .venv/bin/activate`
3.  **Installing Packages**:
    *   `uv pip install <package>`
    *   `uv pip install -r requirements.txt`
4.  **Project Management**:
    *   `uv` respects `pyproject.toml` if present.
    *   It can generate lockfiles: `uv pip compile pyproject.toml -o requirements.txt`

## Common Commands

*   `uv pip install pandas numpy` : Install specific packages
*   `uv pip list` : List installed packages
*   `uv pip sync requirements.txt` : Sync environment with requirements file (removes extraneous packages)
*   `uv cache clean` : Clean the cache

## Integration

*   `uv` automatically detects and uses the active virtual environment (`.venv`).
*   It is significantly faster than `pip` due to its Rust implementation and advanced caching.
