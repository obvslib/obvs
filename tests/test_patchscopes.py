import pytest
from patchscopes_nnsight.patchscopes import SourceContext, TargetContext, Patchscope


# Make a patchscope fixture so we only have to load the model once. (This is slow)
@pytest.fixture(scope="session")
def patchscope():
    source_context = SourceContext(device="cpu")
    target_context = TargetContext.from_source(source_context, max_new_tokens=1)
    return Patchscope(source_context, target_context)


# Make a patchscope fixture so we only have to load the model once. (This is slow)
@pytest.fixture(scope="session")
def patchscope_llama():
    source_context = SourceContext(device="cpu", model_name="meta-llama/Llama-2-70b-hf")
    target_context = TargetContext.from_source(source_context, max_new_tokens=1)
    return Patchscope(source_context, target_context)


class TestPatchscope:
    @staticmethod
    def test_equal_full_patch(patchscope):
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.target.max_new_tokens = 1
        patchscope.get_position_and_layer()

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_equal_full_patch_all_layers(patchscope):
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.get_position_and_layer()

        for i in range(patchscope.n_layers):
            patchscope.run()
            output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
            decoded = patchscope.target_model.tokenizer.decode(output)

            # Assert the target has been patched to think a rat is a cat
            assert "cat" in decoded

    @staticmethod
    def test_equal_single_patch(patchscope):
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.source.position = -1
        patchscope.target.position = -1

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_unequal_single_patch(patchscope):
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.source.position = -1
        patchscope.target.position = -1

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_single_patch_early(patchscope):
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.source.layer = 3
        patchscope.target.layer = 3

        # Get the index of 'cat'
        patchscope.source.position = patchscope.find_in_source(" cat")
        # Patch the first instance of "rat"
        patchscope.target.position = patchscope.find_in_target(" rat")

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_multi_token_generation(patchscope):
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.max_new_tokens = 4

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "a cat is a cat" in "".join(patchscope.full_output())

    @staticmethod
    def test_multi_token_generation_with_patch(patchscope):
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.get_position_and_layer()
        patchscope.target.max_new_tokens = 4

        patchscope.source.layer = 3
        patchscope.target.layer = 3

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "a rat is a cat" in "".join(patchscope.full_output())

    @staticmethod
    def test_justify_gpt(patchscope):
        patchscope.source.prompt = " 1 2 3 4 5 6 7 8 9 0"
        assert len(patchscope.source_tokens) == 10
        tokens_a, tokens_b = patchscope.justify(" 1")

        assert len(tokens_a) == 9
        assert len(tokens_b) == 9

        tokens_a, tokens_b = patchscope.justify(" 1", " 1 2 3")
        assert len(tokens_a) == 9
        assert len(tokens_b) == 9

        tokens_a, tokens_b = patchscope.justify(" 1", bomb=False)
        assert len(tokens_a) == 1
        assert len(tokens_b) == 1

        tokens_a, tokens_b = patchscope.justify(" 1", " 1 2 3", bomb=False)
        assert len(tokens_a) == 3
        assert len(tokens_b) == 3

    @staticmethod
    def test_justify_llama(patchscope_llama):
        patchscope = patchscope_llama
        patchscope.source.prompt = " 1 2 3 4 5 6 7 8 9 0"
        assert len(patchscope.source_tokens) == 21
        tokens_a, tokens_b = patchscope.justify(" 1")

        assert len(tokens_a) == 18
        assert len(tokens_b) == 18

        tokens_a, tokens_b = patchscope.justify(" 1", " 1 2 3")
        assert len(tokens_a) == 18
        assert len(tokens_b) == 18

        tokens_a, tokens_b = patchscope.justify(" 1", bomb=False)
        assert len(tokens_a) == 3
        assert len(tokens_b) == 3

        tokens_a, tokens_b = patchscope.justify(" 1", " 1 2 3", bomb=False)
        assert len(tokens_a) == 7
        assert len(tokens_b) == 7
