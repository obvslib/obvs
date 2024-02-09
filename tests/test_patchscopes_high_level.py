import pytest
from obvspython.patchscopes import SourceContext, TargetContext, Patchscope


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


class TestPatchscope:
    @staticmethod
    def test_equal_full_patch(patchscope):
        """
        If you copy the activations for all tokens, the target becomes the source
        """
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.target.max_new_tokens = 1
        patchscope.get_position()

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_equal_full_patch_all_layers(patchscope):
        """
        This should work actoss all layers
        """
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.get_position()

        for i in range(patchscope.n_layers):
            patchscope.run()
            output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
            decoded = patchscope.target_model.tokenizer.decode(output)

            # Assert the target has been patched to think a rat is a cat
            assert "cat" in decoded

    @staticmethod
    def test_equal_single_patch(patchscope):
        """
        Only patching the last token should do a similar thing, at least at the final layer
        """
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
        """
        Patching only one token, we don't need the two to be the same length
        We can try patching out the last token earlier. (Should this always work? Maybe not right?)
        """
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
        """
        Check we can generate more than one token at the target side
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.max_new_tokens = 4

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "a cat is a cat" in patchscope.full_output()

    @staticmethod
    def test_multi_token_generation_with_patch(patchscope):
        """
        And the patch works with multi-token generation across subsequent tokens
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.get_position()
        patchscope.target.max_new_tokens = 4

        patchscope.source.layer = 3
        patchscope.target.layer = 3

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "a rat is a cat" in patchscope.full_output()

    @staticmethod
    def test_token_identity_prompt(patchscope):
        """
        This is the same as the last setup, but we use a more natural set of prompts.
        """
        patchscope.source.prompt = "it has whiskers and a tail. it domesticated itself. it is a"
        patchscope.target.prompt = "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x"
        patchscope.target.max_new_tokens = 4

        patchscope.source.position = -1
        # Patch the index of "x"
        patchscope.target.position = patchscope.find_in_target(" x")

        # At the end, assume the final token has been loaded with the concept of 'cat'
        patchscope.source.layer = range(patchscope.n_layers)
        # Patch it at every layer
        patchscope.target.layer = range(patchscope.n_layers)

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "x is a cat" in patchscope.full_output()
