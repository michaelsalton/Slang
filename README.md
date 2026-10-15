# Slang Research

Experiments with the [Slang shading language](https://shader-slang.org), driven from Python via
[slangpy](https://slangpy.shader-slang.org).

## Prerequisites

- **A GPU with up-to-date drivers.** slangpy picks a backend automatically: D3D12/Vulkan on
  Windows, Vulkan on Linux, Metal on macOS.
- **[uv](https://docs.astral.sh/uv/)**, which manages Python and the project's virtual environment.
  You don't need to install Python yourself; uv downloads the version pinned in `.python-version`
  (3.12).
- **macOS only:** slangpy runs on Metal and needs Xcode's Metal Toolchain (one-time):

  ```sh
  xcodebuild -downloadComponent MetalToolchain
  ```

## Setup

### 1. Install uv

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal afterwards so `uv` is on your `PATH`, then check it with `uv --version`.
If you're using VS Code's integrated terminal, restart VS Code itself (all windows); opening a
new terminal tab isn't enough.

> **Still getting "uv is not recognized"?** On Windows, uv installs to `%USERPROFILE%\.local\bin`.
> You can add it to `PATH` for the current PowerShell session only:
>
> ```powershell
> $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
> ```

### 2. Create the environment and install dependencies

From the repository root:

```sh
uv sync
```

This creates `.venv/`, installs Python 3.12 if needed, and installs every dependency from
`uv.lock` (slangpy, numpy, matplotlib, and the dev tools pytest, ruff, and ipykernel). It also
installs this project in editable mode, so `slang_research` and `src/app.py` can be imported.
Run `uv sync` again whenever `pyproject.toml` or `uv.lock` changes.

### 3. Activate the virtual environment (optional)

You can skip this step and prefix commands with `uv run` instead (see below). If you'd rather run
`python` and `pytest` directly, activate the environment:

| Shell | Command |
| --- | --- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd | `.venv\Scripts\activate.bat` |
| Git Bash (Windows) | `source .venv/Scripts/activate` |
| macOS / Linux | `source .venv/bin/activate` |

Your prompt should now start with `(slang-research)`. Run `deactivate` to leave the environment.

> **PowerShell says "running scripts is disabled on this system"?** Allow local scripts for your
> user once, then try again:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 4. Verify the install

```sh
uv run pytest
```

If the tests pass, slangpy found your GPU and compiled the shaders.

## Running

Every command below can be run in either of two ways:

- **With `uv run`** (no activation needed): `uv run python experiments/hello.py`
- **With the environment activated**: `python experiments/hello.py`

| Command | What it does |
| --- | --- |
| `uv run python experiments/hello.py` | Runs the hello-world shader and writes `outputs/checkerboard.png` |
| `uv run python experiments/neural_image.py` | Opens a window that trains a small MLP on the GPU to reproduce `images/slangstars.png` |
| `uv run pytest` | Runs the shader tests |
| `uv run ruff check . && uv run ruff format .` | Lints and formats the code |

The first run of an experiment compiles its shaders, which can take a while.

### Neural image window controls

| Key | Action |
| --- | --- |
| `Esc` | Close the window |
| `F1` | Open the current output in [tev](https://github.com/Tom94/tev), if it's installed |
| `F2` | Save a screenshot to `screenshot.png` |

The window shows the reference image, the network's current reconstruction, and the per-pixel
loss side by side. The mean loss is printed to the terminal every frame.

### Notebooks

Open a notebook in `notebooks/` with VS Code or Jupyter and pick this project's `.venv` as the
kernel. `ipykernel` is already installed by `uv sync`.

## Layout

| Path | Purpose |
| --- | --- |
| `shaders/` | `.slang` modules (on the include path, so `import foo;` works between them) |
| `src/slang_research/` | Shared Python helpers (`create_device`, `load_module`) |
| `src/app.py` | `App` window class used by interactive experiments |
| `experiments/` | Standalone experiment scripts |
| `images/` | Input images for experiments |
| `notebooks/` | Jupyter notebooks (kernel: this project's `.venv`) |
| `tests/` | pytest checks of shader behaviour against numpy |
| `outputs/` | Generated images/data (git-ignored) |

## Calling a Slang function from Python

```python
import numpy as np
from slang_research import load_module

m = load_module("hello.slang")
m.add(np.ones(4, np.float32), 2.0, _result="numpy")
```
