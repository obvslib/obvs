# obvs

[![CI](https://github.com/obvslib/obvs/actions/workflows/main.yaml/badge.svg)](https://github.com/obvslib/obvs/actions/workflows/main.yaml)

Making Transformers Obvious

## Project cheatsheet

-   **pre-commit:** `pre-commit run --all-files`
-   **pytest:** `pytest` or `pytest -s`
-   **coverage:** `coverage run -m pytest` or `coverage html`
-   **poetry sync:** `poetry install --no-root --sync`
-   **updating requirements:** see [docs/updating_requirements.md](docs/updating_requirements.md)

## Initial project setup

1. See [docs/getting_started.md](docs/getting_started.md) or [docs/quickstart.md](docs/quickstart.md)
   for how to get up & running.
2. Check [docs/project_specific_setup.md](docs/project_specific_setup.md) for project specific setup.
3. See [docs/using_poetry.md](docs/using_poetry.md) for how to update Python requirements using
   [Poetry](https://python-poetry.org/).
4. See [docs/detect_secrets.md](docs/detect_secrets.md) for more on creating a `.secrets.baseline`
   file using [detect-secrets](https://github.com/Yelp/detect-secrets).
