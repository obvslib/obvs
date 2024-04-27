"""
The `obvs` library is a Python package that provides tools and utilities for analyzing and interpreting language models using the patchscope framework. It offers a range of functionalities to probe and understand the internal representations and behaviors of language models at different layers and positions.

1. `obvs` directory:
   - The `obvs` directory contains the main components of the library, including the `patchscope`, `patchscope_base`, `lenses`, `logging`, and `metrics` modules.

2. `patchscope` and `patchscope_base`:
   - The `patchscope` module implements the core functionality of the patchscope framework, which allows for mapping and patching representations between different language models.
   - The `patchscope_base` module serves as an abstract base class for the patchscope framework, providing common functionality and abstractions.
   - Together, these modules enable the analysis and interpretation of language models by mapping representations from a source model to a target model and studying the effects of interventions.

3. `lenses`:
   - The `lenses` module provides various lenses for analyzing language models, such as `TokenIdentity`, `BaseLogitLens`, `PatchscopeLogitLens`, and `ClassicLogitLens`.
   - Lenses are techniques that allow for probing and understanding the internal representations and behaviors of language models at different layers and positions.
   - The module includes methods for running lens analyses, computing metrics like surprisal and precision@1, and visualizing the results using heatmaps and plots.

4. `logging`:
   - The `logging` module provides utility functions and classes for configuring and customizing the logging behavior in the application.
   - It includes a custom logging handler (`TqdmLoggingHandler`) that integrates with the tqdm progress bar library, allowing log messages to be displayed alongside the progress bar.
   - The module also configures a specific logger named "patchscope" with a file handler that logs messages to a file named "experiments.log".

5. `metrics`:
   - The `metrics` module defines evaluation metrics for language modeling tasks.
   - It includes classes like `PrecisionAtK` and `Surprisal` for computing precision@k and surprisal metrics, respectively.
   - These metrics can be used to assess the performance of language models and evaluate the effectiveness of different interpretation techniques.

6. `scripts` directory:
   - The `scripts` directory contains a collection of scripts that serve as a cookbook for reproducing standard results using the `obvs` library and the `patchscope` framework.
   - The scripts demonstrate how to use different lenses and techniques to analyze and interpret language models.
   - Some notable scripts include:
     - `activation_patching_ioi`: Uses activation patching to study indirect object identification (IOI) on gpt2-small.
     - `future_lens`: Generates the future lens at a single position.
     - `generate_next_token_prediction_data`: Generates data for next token prediction tasks.
     - `replicate_figure_2`: Replicates Figure 2 from a specific research paper.
     - `reproduce_logitlens_results`: Reproduces the results of the original logitlens blog post.
     - `token_identity_prompts` and `token_identity`: Demonstrate the usage of the token identity lens.

Overall, the `obvs` library provides a comprehensive set of tools and utilities for analyzing and interpreting language models using the patchscope framework. It offers a range of lenses, metrics, and visualization techniques to study the internal representations and behaviors of models at different layers and positions. The included scripts serve as practical examples and a starting point for conducting interpretability experiments and reproducing standard results.
"""
from __future__ import annotations

__version__ = "0.1.0"
