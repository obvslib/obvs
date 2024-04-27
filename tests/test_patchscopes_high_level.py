from __future__ import annotations

import pytest


class TestPatchscope:
    @staticmethod
    def test_equal_full_patch(patchscope):
        """
        If you copy the activations for all tokens, the target becomes the source
        """
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.source.layer = -1
        patchscope.target.layer = -1

        # This configuration will set it to take all tokens
        patchscope.source.position = None
        patchscope.target.position = None

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_equal_full_patch_all_layers(patchscope):
        """
        This should work across all layers
        """
        patchscope.source.prompt = "a dog is a dog. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1

        for i in range(patchscope.n_layers):
            patchscope.source.level = i
            patchscope.target.level = i

            patchscope.run()

            output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
            decoded = patchscope.tokenizer.decode(output)

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
        patchscope.source.layer = -1
        patchscope.target.layer = -1

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.target_model.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    def test_unequal_single_patch(patchscope):
        """
        Patching only one token, we don't need the two to be the same length
        At the final layer this should work.
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat is a"
        patchscope.target.prompt = "a dog is a dog. a rat is a"
        patchscope.target.max_new_tokens = 1
        patchscope.source.position = -1
        patchscope.target.position = -1
        patchscope.source.layer = -1
        patchscope.target.layer = -1

        patchscope.run()
        output = patchscope._target_outputs[0].value.argmax(dim=-1)[-1].tolist()
        decoded = patchscope.tokenizer.decode(output)

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
        decoded = patchscope.tokenizer.decode(output)

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in decoded

    @staticmethod
    @pytest.mark.skip(reason="This doesn't work")
    def test_multi_token_generation(patchscope):
        """
        Check we can generate more than one token at the target side
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = patchscope.source.prompt
        patchscope.target.max_new_tokens = 4

        patchscope.run()

        assert "a cat is a cat" in patchscope.full_output()

    @staticmethod
    @pytest.mark.skip(reason="This doesn't work")
    def test_multi_token_generation_with_patch(patchscope):
        """
        And the patch works with multi-token generation across subsequent tokens
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.target.max_new_tokens = 4

        patchscope.source.layer = 3
        patchscope.target.layer = 3

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "a rat is a cat" in patchscope.full_output()

    @staticmethod
    @pytest.mark.skip(reason="This doesn't work")
    def test_multi_token_generation_with_different_lengths_single_patch(patchscope):
        """
        And the patch works with multi-token generation across subsequent tokens
        """
        patchscope.source.prompt = "frog"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = -1
        patchscope.target.position = -1
        patchscope.target.max_new_tokens = 4

        patchscope.source.layer = 3
        patchscope.target.layer = 3

        patchscope.run()

        # Assert the target has been patched to think a rat is a cat
        assert "cat" in patchscope.full_output()

    @staticmethod
    @pytest.mark.skip(reason="This doesn't work")
    def test_soft_prompt(patchscope):
        soft_prompt = None
        with patchscope.source_model.trace("a dog is a dog. a cat is a", remote=False):
            soft_prompt = patchscope.source_model.transformer.wte.output.save()

        patchscope.source.prompt = soft_prompt.value
        patchscope.source.position = -1
        patchscope.source.layer = -1

        patchscope.target.prompt = " ".join(
            "_" * soft_prompt.shape[1],
        )  # works for gpt2 & gptj, not sure about others
        patchscope.target.position = -1
        patchscope.target.layer = -1
        patchscope.target.max_new_tokens = 4

        patchscope.run()

        assert "cat" in patchscope.output()[-1]

    @staticmethod
    @pytest.mark.skip(reason="This doesn't work")
    def test_token_identity_prompt_early(patchscope):
        """
        This is the same as the last setup, but we use a more natural set of prompts.
        THIS DOESNT WORK :'(
        """
        patchscope.source.prompt = (
            "it has whiskers and a tail. it domesticated itself. it is a species of"
        )
        patchscope.target.prompt = (
            "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x is"
        )
        patchscope.target.max_new_tokens = 4

        # Take the final token from the source
        patchscope.source.position = -1
        # Patch the index of "x"
        patchscope.target.position = patchscope.find_in_target(" x")

        # At the end, assume the final token has been loaded with the concept of 'cat'
        patchscope.source.layer = -1
        # Patch it at an early layer
        patchscope.target.layer = 3

        patchscope.run()

        # Assert the target has been patched to think about a cat
        assert "cat" in patchscope.full_output()

    @staticmethod
    @pytest.mark.skip(reason="This doesn't work")
    def test_token_identity_prompt(patchscope):
        """
        This is the same as the last setup, but we use a more natural set of prompts.
        THIS DOESNT WORK :'(
        """
        patchscope.source.prompt = (
            "it has whiskers and a tail. it domesticated itself. it is a species of"
        )
        patchscope.target.prompt = (
            "bat is bat; 135 is 135; hello is hello; black is black; shoe is shoe; x is"
        )
        patchscope.target.max_new_tokens = 4

        # Take the final token from the source
        patchscope.source.position = -1
        # Patch the index of "x"
        patchscope.target.position = patchscope.find_in_target(" x")

        # At the end, assume the final token has been loaded with the concept of 'cat'
        patchscope.source.layer = -1
        # Patch it at the last layer
        patchscope.target.layer = -1

        patchscope.run()

        # Assert the target has been patched to think about a cat
        assert "cat" in patchscope.full_output()

    @staticmethod
    @pytest.mark.skip(reason="This doesn't work")
    def test_over(patchscope):
        """
        Test the over method
        """
        patchscope.source.prompt = "a dog is a dog. a rat is a rat. a cat"
        patchscope.target.prompt = "a dog is a dog. a bat is a bat. a rat"
        patchscope.source.position = None
        patchscope.target.position = None
        patchscope.target.max_new_tokens = 2

        values = list(patchscope.over(range(2), range(4)))
        # Its a layer x layer list
        assert len(values) == 8
        # With the outputs of two generations
        assert len(values[0]) == 2
        # The first of which is the length of the target tokens
        assert values[0][0].shape[0] == len(patchscope.target_tokens)
        # And the second has length 1
        assert values[0][1].shape[0] == 1
