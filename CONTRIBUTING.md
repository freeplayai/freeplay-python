# Contributing to the Freeplay Python SDK

Thank you for your interest in contributing to the Freeplay Python SDK! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up your development environment:

```bash
# Install uv if you haven't already
brew install uv

# Run the setup script
./devenv.sh
```

## Development

### Running Tests

```bash
uv run pytest
```

### Linting and Type Checking

```bash
uv run ruff check .
uv run mypy .
uv run pyright
```

## Pull Request Process

1. Create a new branch for your feature or fix
2. Make your changes
3. Ensure all tests pass and linting is clean
4. Submit a pull request with a clear description of the changes

## Code of Conduct

Please be respectful and constructive in all interactions. We're all here to build great software together.

## Questions?

If you have questions, feel free to open an issue or reach out to us at support@freeplay.ai.

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.

