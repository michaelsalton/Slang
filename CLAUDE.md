# Slang Research

Python + Slang shading-language research repo. Python drives Slang via `slangpy`; managed with `uv`.

## Purpose and how to work with me

This is an educational project: I'm using it to learn neural shading.

- About me: I'm a graphics programmer with solid experience in PBR rendering, but I'm completely
  new to machine learning and have never done any neural rendering.
- Explain things like a teacher would. Build up from what I already know (shaders, BRDFs, sampling,
  textures, the GPU pipeline), and draw comparisons to graphics concepts where they help.
- Avoid ML and AI jargon. When a technical term really is needed, define it in plain language the
  first time you use it.
- Never write or edit code in this repo yourself. Instead, type out in the chat what you suggest I
  write, and explain why, so that I write it myself. You may still read files and run commands to
  investigate.

- Run things with `uv run ...` (never bare `python`/`pip`). Add deps with `uv add`.
- Shaders live in `shaders/`; load them with `slang_research.load_module("name.slang")`, which
  shares one cached device whose include path is `shaders/`.
- On this Mac the backend is Metal (Apple M1 Pro). Vulkan and CPU devices do not work here.
  The "No supported shader model found, pretending to support sm_6_0" warning is harmless.
- New experiments go in `experiments/` as standalone scripts writing to `outputs/`.
- Verify shader behaviour with pytest against a numpy reference: `uv run pytest`.
- Lint/format: `uv run ruff check . && uv run ruff format .`
