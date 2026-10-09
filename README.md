# Slang Research

Experiments with the [Slang shading language](https://shader-slang.org), driven from Python via
[slangpy](https://slangpy.shader-slang.org).

## Setup

Requires [uv](https://docs.astral.sh/uv/). On macOS, slangpy runs on Metal and needs Xcode's
Metal Toolchain (one-time):

```sh
xcodebuild -downloadComponent MetalToolchain
uv sync
```

## Layout

| Path | Purpose |
| --- | --- |
| `shaders/` | `.slang` modules (on the include path, so `import foo;` works between them) |
| `src/slang_research/` | Shared Python helpers (`create_device`, `load_module`) |
| `experiments/` | Standalone experiment scripts |
| `notebooks/` | Jupyter notebooks (kernel: this project's `.venv`) |
| `tests/` | pytest checks of shader behaviour against numpy |
| `outputs/` | Generated images/data (git-ignored) |

## Usage

```sh
uv run python experiments/hello.py   # writes outputs/checkerboard.png
uv run pytest                         # run shader tests
uv run ruff check . && uv run ruff format .
```

Calling a Slang function from Python:

```python
from slang_research import load_module

m = load_module("hello.slang")
m.add(np.ones(4, np.float32), 2.0, _result="numpy")
```
