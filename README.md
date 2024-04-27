# obvs: A Python Library for Analyzing and Interpreting Language Models

The `obvs` library is a powerful Python package that provides a comprehensive set of tools and utilities for analyzing and interpreting language models using the patchscope framework. It offers a range of functionalities to probe and understand the internal representations and behaviors of language models at different layers and positions.

With `obvs`, you can easily investigate how language models process and generate text, gain insights into their inner workings, and visualize the results using various techniques such as heatmaps and plots. Whether you are a researcher, data scientist, or language model enthusiast, `obvs` empowers you to conduct interpretability experiments and reproduce standard results with ease.

## Installation

To install the `obvs` library, you can use pip:

```
pip install obvs
```

Make sure you have Python >=3.10 installed on your system.

## Entity Resolution Example.

Here's a simple example demonstrating how to use the `TokenIdentity` lens from the `obvs` library:

```python
from obvs.lenses import TokenIdentity

token_identity = TokenIdentity(source_prompt="The quick brown fox", model_name="gpt2", source_phrase="quick brown")
token_identity.run().compute_surprisal("fox").visualize()
```

This code snippet creates an instance of the `TokenIdentity` lens, specifying the source prompt, model name, and source phrase. It then runs the lens analysis, computes the surprisal for the word "fox", and visualizes the results.

## Activation Patching Example.

"The Collosseum is in the city of Paris" activation patching example.

```python
from obvs.patchscope import Patchscope, SourceContext, TargetContext

MODEL="gpt2"
source=SourceContext(
    model_name=MODEL,
    prompt="The Eiffel Tower is in the city of",
    layer=10,
    position=9,
)

target=TargetContext(
    model_name=MODEL,
    prompt="The Colosseum is in the city of",
    layer=10,
    position=9,
    max_new_tokens=1
)

patchscope=Patchscope(source, target)
patchscope.run()

print(patchscope.full_output())
```

For more examples and usage, please refer to the [tutorials](https://github.com/obvslib/obvs/tree/main/tutorials), [documentation](https://obvs.rtfd.io/), and [PyPI](https://pypi.org/project/obvs/).

## Development setup

To set up the development environment for `obvs`, follow these steps:

1. Clone the repository:

    ```
    git clone https://github.com/yourname/obvs.git
    ```

2. Install the development dependencies:

    ```
    poetry install --no-root --sync
    ```

3. Run the test suite:
    ```
    pytest tests/
    ```

Make sure you have Python 3.10 or above and the required dependencies installed.

## Release History

-   0.1.2
    -   Initial release of the `obvs` library
    -   Includes `patchscope`, `patchscope_base`, `lenses`, `logging`, and `metrics` modules
    -   Provides a collection of scripts for reproducing standard results

## Meta

For assistance, reach out to Jamie Coombes – www.linkedin.com/in/
jamiecoombes – obvslib@protonmail.com

Distributed under the MIT license. See `LICENSE` for more information.

## Contributing

We welcome contributions to the `obvs` library! See `CONTRIBUTING` for more information.
