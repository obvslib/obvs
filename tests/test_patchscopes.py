from obvspython.patchscopes import SourceContext, TargetContext, Patchscope

import torch


class TestContext:
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

        # Test with a random torch tensor
        tensor = torch.tensor([1, 2, 3])
        assert target.mapping_function(tensor) == tensor

        target = TargetContext("target", mapping_function=lambda x: x + 1)
        assert target.mapping_function(1) == 2
        assert target.mapping_function(tensor) == tensor + 1


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

        assert patchscope.souce.layer == -1
        assert patchscope.target.layer == -1

    @staticmethod
    def test_patchscope_map():
        source = SourceContext("source")
        target = TargetContext("target")
        patchscope = Patchscope(source, target)
