# LLM Explorer

This project aims to investigate how Large Language Models (LLMs) explore in Reinforcement Learning (RL) settings. The primary questions are:
- How is LLM-driven exploration different from optimal exploration strategies?
- Can prompting LLMs with RL theory enhance their exploration capabilities?

## Project Structure

- `src/llm_explorer/`: Contains the core Python package.
  - `rlllm_utils.py`: Implements Multi-Armed Bandit environments, Thompson Sampling, and an LLM-based agent.
  - `llm.py`: Provides a wrapper for interacting with a generative AI model.
- `tests/`: Contains unit tests for the package.
- `docs/`: Contains documentation, including a placeholder for a literature review (`lit_review.md`).
- `pyproject.toml`: Defines package metadata, dependencies, and build system configuration.

## Installation

To install the package for development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/yourusername/llm_explorer.git # TODO: Replace with your repo URL
cd llm_explorer
pip install -e .[dev]
```

The `[dev]` option includes dependencies needed for testing (like `pytest`).

## Usage

(Details on how to run experiments or use the library will be added here.)

## Contributing

(Details on how to contribute will be added here.)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. (Note: A `LICENSE` file should be added if one doesn't exist).
