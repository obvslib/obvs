from __future__ import annotations

import pytest

from obvspython.patchscope import Patchscope, SourceContext, TargetContext


# Make a patchscope fixture so we only have to load the model once. (This is slow)
@pytest.fixture(scope="session")
def patchscope():
    source_context = SourceContext(device="cpu")
    target_context = TargetContext.from_source(source_context, max_new_tokens=1)
    return Patchscope(source_context, target_context)


# Make a patchscope fixture so we only have to load the model once. (This is slow)
@pytest.fixture(scope="session")
def patchscope_llama():
    # Don't worry, we just use the tokenizer, you won't need to download the full llama2 model ;)
    source_context = SourceContext(device="cpu", model_name="meta-llama/Llama-2-7b-hf")
    target_context = TargetContext.from_source(source_context, max_new_tokens=1)
    return Patchscope(source_context, target_context)
