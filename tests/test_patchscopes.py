from __future__ import annotations

import numpy as np
import pytest
import torch

from obvs.patchscope import Patchscope, SourceContext, TargetContext


class TestContext:
    STUB_EMBEDDING = torch.ones((1, 2))

    @staticmethod
    def test_source_context_init():
        source = SourceContext("source")
        assert source.prompt == "source"

    @staticmethod
    def test_target_context_init():
        target = TargetContext("target")
        assert target.prompt == "target"

    @staticmethod
    def test_target_context_default_mapping():
        target = TargetContext("target")
        assert target.mapping_function(1) == 1
        assert target.mapping_function(None) is None

        # Test with a random torch tensor using random numbers
        tensor = torch.tensor(np.random.rand(3, 3))
        assert target.mapping_function(tensor).equal(tensor)

        target = TargetContext("target", mapping_function=lambda x: x + 1)
        assert target.mapping_function(1) == 2
        assert target.mapping_function(tensor).equal(tensor + 1)

    def test_can_only_have_prompt_or_embedding(self):
        with pytest.raises(ValueError):
            SourceContext(prompt="foo", embedding=self.STUB_EMBEDDING)

        with pytest.raises(ValueError):
            TargetContext(prompt="foo", embedding=self.STUB_EMBEDDING)

        # Should not throw exceptions
        SourceContext(prompt="foo")
        TargetContext(prompt="foo")

        SourceContext(embedding=self.STUB_EMBEDDING)
        TargetContext(embedding=self.STUB_EMBEDDING)

    def test_embedding_dimensions_must_be_two(self):
        SourceContext(embedding=self.STUB_EMBEDDING)

        with pytest.raises(ValueError):
            SourceContext(embedding=torch.ones((1,)))

        with pytest.raises(ValueError):
            SourceContext(embedding=torch.ones((1,2,3)))


class TestPatchscope:
    @staticmethod
    def test_patchscope_init():
        source = SourceContext("source")
        target = TargetContext("target")
        patchscope = Patchscope(source, target)
        assert patchscope.source == source
        assert patchscope.target == target
        assert patchscope.source_model
        assert patchscope.target_model

        assert patchscope.source.position == range(len(patchscope.tokenizer.encode("source")))
        assert patchscope.target.position == range(len(patchscope.tokenizer.encode("target")))

        assert patchscope.source.layer == -1
        assert patchscope.target.layer == -1

    @staticmethod
    def test_patchscope_map():
        source = SourceContext("source")
        target = TargetContext("target")
        patchscope = Patchscope(source, target)

        tensor = torch.tensor(np.random.rand(3, 3))
        patchscope._source_hidden_state = tensor
        patchscope.map()
        assert patchscope._mapped_hidden_state.equal(tensor)

    @staticmethod
    def test_patchscope_map_transpose():
        source = SourceContext("source")
        target = TargetContext("target")
        target.mapping_function = lambda x: torch.transpose(x, 0, 1)
        patchscope = Patchscope(source, target)

        tensor = torch.tensor(np.random.rand(3, 3))
        patchscope._source_hidden_state = tensor
        patchscope.map()
        assert patchscope._mapped_hidden_state.equal(tensor.T)

    @staticmethod
    def test_source_tokens(patchscope):
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        assert patchscope.source_tokens == patchscope.tokenizer.encode("a dog is a dog. a cat is a")

    @staticmethod
    def test_source_forward_pass_creates_hidden_state(patchscope):
        patchscope.source.prompt = "a dog is a dog. a cat is a"

        # This will set position to all tokens
        patchscope.source.position = None
        patchscope.init_positions()

        patchscope.source.layer = 0
        patchscope.source_forward_pass()

        assert patchscope._source_hidden_state.value.shape[0] == 1  # Batch size, always 1
        assert patchscope._source_hidden_state.value.shape[1] == len(
            patchscope.source_tokens,
        )  # Number of tokens
        assert (
            patchscope._source_hidden_state.value.shape[2]
            == patchscope.source_model.transformer.embed_dim
        )  # Embedding dimension

        patchscope.source.prompt = "a dog is a dog"
        patchscope.init_positions(force=True)
        patchscope.source_forward_pass()
        assert patchscope._source_hidden_state.value.shape[1] == len(
            patchscope.source_tokens,
        )  # Number of tokens
