from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class PatchscopeBase(ABC):
    """
    A base class with lots of helper functions
    """

    def get_model_specifics(self, model_name):
        """
        Get the model specific attributes.
        The following works for gpt2, llama2 and mistral models.
        """
        if "gpt" in model_name:
            return "transformer", "h", "attn", "c_proj"
        if "mamba" in model_name:
            return "backbone", "layers", None, None
        return "model", "layers", "attention", "heads"

    @abstractmethod
    def source_forward_pass(self) -> None:
        pass

    @abstractmethod
    def map(self) -> None:
        pass

    @abstractmethod
    def target_forward_pass(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @property
    def _source_position(self) -> Sequence[int]:
        return (
            self.source.position
            if self.source.position is not None
            else range(len(self.source_token_ids))
        )

    @property
    def _target_position(self) -> Sequence[int]:
        return (
            self.target.position
            if self.target.position is not None
            else range(len(self.target_token_ids))
        )

    @property
    def source_token_ids(self) -> list[int]:
        """
        Return the source tokens
        """
        return self.tokenizer.encode(self.source.text_prompt)

    @property
    def target_token_ids(self) -> list[int]:
        """
        Return the target tokens
        """
        return self.tokenizer.encode(self.target.text_prompt)

    @property
    def source_tokens(self) -> list[str]:
        """
        Return the input to the source model
        """
        return [self.tokenizer.decode(token) for token in self.source_token_ids]

    @property
    def target_tokens(self) -> list[str]:
        """
        Return the input to the target model
        """
        return [self.tokenizer.decode(token) for token in self.target_token_ids]

    def top_k_tokens(self, k: int = 10) -> list[str]:
        """
        Return the top k tokens from the target model
        """
        token_ids = self._target_outputs[0].value[self.target.position, :].topk(k).indices.tolist()
        return [self.tokenizer.decode(token_id) for token_id in token_ids]

    def top_k_logits(self, k: int = 10) -> list[int]:
        """
        Return the top k logits from the target model
        """
        return self._target_outputs[0].value[self.target.position, :].topk(k).values.tolist()

    def top_k_probs(self, k: int = 10) -> list[float]:
        """
        Return the top k probabilities from the target model
        """
        # FIXME: broken, returns a list of [1.0]
        logits = self.top_k_logits(k)
        return [torch.nn.functional.softmax(torch.tensor(logit), dim=-1).item() for logit in logits]

    def logits(self) -> torch.Tensor:
        """
        Return the logits from the target model (size [pos, d_vocab])
        """
        return self._target_outputs[0].value[:, :]

    def probabilities(self) -> torch.Tensor:
        """
        Return the probabilities from the target model (size [pos, d_vocab])
        """
        return torch.softmax(self.logits(), dim=-1)

    def output(self) -> list[str]:
        """
        Return the generated output from the target model
        """
        token_ids = self.logits().argmax(dim=-1)
        return [self.tokenizer.decode(token_id) for token_id in token_ids]

    def _output_token_ids(self) -> list[int]:
        tensors_list = [tensor_proxy.value for tensor_proxy in self._target_outputs]
        tokens = torch.cat(tensors_list, dim=0)
        return tokens.argmax(dim=-1).tolist()

    def llama_output(self) -> list[str]:
        """
        For llama, if you don't decode them all together, they don't add the spaces.
        """
        tokens = self._output_token_ids()
        return self.tokenizer.decode(tokens)

    def full_output_tokens(self) -> list[str]:
        """
        Return the generated output from the target model
        This is a bit hacky. Its not super well supported. I have to concatenate
        all the inputs and add the input tokens to them.
        """
        token_ids = self._output_token_ids()

        input_token_ids = self.tokenizer.encode(self.target.text_prompt)
        token_ids.insert(0, " ")
        token_ids[: len(input_token_ids)] = input_token_ids
        return [self.tokenizer.decode(token_id) for token_id in token_ids]

    def full_output(self) -> str:
        """
        Return the generated output from the target model
        This is a bit hacky. Its not super well supported.
        I have to concatenate all the inputs and add the input tokens to them.
        """
        return "".join(self.full_output_tokens())

    def find_in_source(self, substring: str) -> int:
        """
        Find the position of the substring tokens in the source prompt

        Note: only works if substring's tokenization happens to match that of
        the source prompt's tokenization
        """
        position, _ = self.source_position_tokens(substring)
        return position

    def source_position_tokens(self, substring: str) -> tuple[int, list[int]]:
        """
        Find the position of a substring in the source prompt, and return the
        substring tokenized

        NB: The try: except block handles the difference between gpt2 and llama
        tokenization. Perhaps this can be better dealt with a seperate tokenizer
        class that handles the differences between the tokenizers. There are a
        few subtleties there, and tokenizing properly is important for getting
        the best out of your model.
        """
        if substring not in self.source.text_prompt:
            raise ValueError(
                f"Substring {substring} could not be found in {self.source.text_prompt}",
            )

        try:
            token_ids = self.tokenizer.encode(substring, add_special_tokens=False)
            return self.source_token_ids.index(token_ids[0]), token_ids
        except ValueError:
            token_ids = self.tokenizer.encode(" " + substring, add_special_tokens=False)
            return self.source_token_ids.index(token_ids[0]), token_ids

    def find_in_target(self, substring: str) -> int:
        """
        Find the position of the substring tokens in the target prompt

        Note: only works if substring's tokenization happens to match that of
        the target prompt's tokenization
        """
        position, _ = self.target_position_tokens(substring)
        return position

    def target_position_tokens(self, substring) -> tuple[int, list[int]]:
        """
        Find the position of a substring in the target prompt, and return the
        substring tokenized

        NB: The try: except block handles the difference between gpt2 and llama
        tokenization. Perhaps this can be better dealt with a seperate tokenizer
        class that handles the differences between the tokenizers. There are a
        few subtleties there, and tokenizing properly is important for getting
        the best out of your model.
        """
        if substring not in self.target.text_prompt:
            raise ValueError(
                f"Substring {substring} could not be found in {self.target.text_prompt}",
            )

        try:
            token_ids = self.tokenizer.encode(substring, add_special_tokens=False)
            return self.target_token_ids.index(token_ids[0]), token_ids
        except ValueError:
            token_ids = self.tokenizer.encode(" " + substring, add_special_tokens=False)
            return self.target_token_ids.index(token_ids[0]), token_ids

    @property
    def n_layers(self) -> int:
        return self.n_layers_target

    @property
    def n_layers_source(self) -> int:
        return len(
            getattr(getattr(self.source_model, self.target_base_name), self.target_layer_name),
        )

    @property
    def n_layers_target(self) -> int:
        return len(
            getattr(getattr(self.target_model, self.target_base_name), self.target_layer_name),
        )

    def compute_precision_at_1(self, estimated_probs: torch.Tensor, true_token_index):
        """
        Compute Precision@1 metric. From the outputs of the target (patched) model
        (estimated_probs) against the output of the source model, aka the 'true' token.
        Args:
        - estimated_probs: The estimated probabilities for each token as a torch.Tensor.
        - true_token_index: The index of the true token in the vocabulary.
        Returns:
        - precision_at_1: Precision@1 metric result.

        This is the evaluation method of the token identity from patchscopes:
        https://arxiv.org/abs/2401.06102
        Its used for running an evaluation over large datasets.
        """
        predicted_token_index = torch.argmax(estimated_probs)
        precision_at_1 = 1 if predicted_token_index == true_token_index else 0
        return precision_at_1

    def compute_surprisal(self, estimated_probs: torch.Tensor, true_token_index):
        """
        Compute Surprisal metric. From the outputs of the target (patched) model
        (estimated_probs) against the output of the source model, aka the 'true'
        token.

        Args:
        - estimated_probs: The estimated probabilities for each token as a
        torch.Tensor.
        - true_token_index: The index of the true token in the vocabulary.

        Returns:
        - surprisal: Surprisal metric result.
        """
        # For now, just compute the surprisal of the first element. We'll need to improve this.
        if isinstance(true_token_index, list):
            true_token_index = true_token_index[0]
        # To avoid log(0) issues, add a small constant to the probabilities
        estimated_probs = torch.clamp(estimated_probs, min=1e-12)
        surprisal = -torch.log(estimated_probs[true_token_index])
        return surprisal.item()
