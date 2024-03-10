from __future__ import annotations

from abc import ABC, abstractmethod

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
            return "transformer", "h"
        if "mamba" in model_name:
            return "backbone", "layers"
        return "model", "layers"

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
    def source_tokens(self):
        """
        Return the source tokens
        """
        return self.tokenizer.encode(self.source.prompt)

    @property
    def target_tokens(self):
        """
        Return the target tokens
        """
        return self.tokenizer.encode(self.target.prompt)

    @property
    def source_words(self):
        """
        Return the input to the source model
        """
        return [self.tokenizer.decode(token) for token in self.source_tokens]

    @property
    def target_words(self):
        """
        Return the input to the target model
        """
        return [self.tokenizer.decode(token) for token in self.target_tokens]

    def init_positions(self, force=False):
        if self.source.position is None or force:
            # If no position is specified, take them all
            self.source.position = range(len(self.source_tokens))

        if self.target.position is None or force:
            self.target.position = range(len(self.target_tokens))

    def top_k_tokens(self, k=10):
        """
        Return the top k tokens from the target model
        """
        tokens = self._target_outputs[0].value[self.target.position, :].topk(k).indices.tolist()
        return [self.tokenizer.decode(token) for token in tokens]

    def top_k_logits(self, k=10):
        """
        Return the top k logits from the target model
        """
        return self._target_outputs[0].value[self.target.position, :].topk(k).values.tolist()

    def top_k_probs(self, k=10):
        """
        Return the top k probabilities from the target model
        """
        logits = self.top_k_logits(k)
        return [torch.nn.functional.softmax(torch.tensor(logit), dim=-1).item() for logit in logits]

    def logits(self):
        """
        Return the logits from the target model
        """
        return self._target_outputs[0].value[:, :]

    def probabilities(self):
        """
        Return the probabilities from the target model
        """
        return torch.softmax(self.logits(), dim=-1)

    def output(self):
        """
        Return the generated output from the target model
        """
        tokens = self.logits().argmax(dim=-1)
        return [self.tokenizer.decode(token) for token in tokens]

    def _output_tokens(self):
        tensors_list = [self._target_outputs[i].value for i in range(len(self._target_outputs))]
        tokens = torch.cat(tensors_list, dim=0)
        return tokens.argmax(dim=-1).tolist()

    def llama_output(self):
        """
        For llama, if you don't decode them all together, they don't add the spaces.
        """
        tokens = self._output_tokens()
        return self.tokenizer.decode(tokens)

    def full_output_words(self):
        """
        Return the generated output from the target model
        This is a bit hacky. Its not super well supported. I have to concatenate all the inputs and add the input tokens to them.
        """
        tokens = self._output_tokens()

        input_tokens = self.tokenizer.encode(self.target.prompt)
        tokens.insert(0, " ")
        tokens[: len(input_tokens)] = input_tokens
        return [self.tokenizer.decode(token) for token in tokens]

    def full_output(self):
        """
        Return the generated output from the target model
        This is a bit hacky. Its not super well supported.
        I have to concatenate all the inputs and add the input tokens to them.
        """
        return "".join(self.full_output_words())

    def find_in_source(self, substring):
        """
        Find the position of the substring tokens in the source prompt
        """
        position, _ = self.source_position_tokens(substring)
        return position

    def source_position_tokens(self, substring):
        """
        Find the position of a substring in the source prompt, and return the substring tokenized

        NB: The try: except block handles the difference between gpt2 and llama
        tokenization. Perhaps this can be better dealt with a seperate tokenizer
        class that handles the differences between the tokenizers. There are a
        few subtleties there, and tokenizing properly is important for getting
        the best out of your model.
        """
        if substring not in self.source.prompt:
            raise ValueError(f"{substring} not in {self.source.prompt}")
        try:
            tokens = self.tokenizer.encode(substring, add_special_tokens=False)
            return self.source_tokens.index(tokens[0]), tokens
        except ValueError:
            tokens = self.tokenizer.encode(" " + substring, add_special_tokens=False)
            return self.source_tokens.index(tokens[0]), tokens

    def find_in_target(self, substring):
        """
        Find the position of the substring tokens in the target prompt
        """
        position, _ = self.target_position_tokens(substring)
        return position

    def target_position_tokens(self, substring):
        """
        Find the position of a substring in the target prompt, and return the substring tokenized

        NB: The try: except block handles the difference between gpt2 and llama
        tokenization. Perhaps this can be better dealt with a seperate tokenizer
        class that handles the differences between the tokenizers. There are a
        few subtleties there, and tokenizing properly is important for getting
        the best out of your model.
        """
        if substring not in self.target.prompt:
            raise ValueError(f"{substring} not in {self.target.prompt}")
        try:
            tokens = self.tokenizer.encode(substring, add_special_tokens=False)
            return self.target_tokens.index(tokens[0]), tokens
        except ValueError:
            tokens = self.tokenizer.encode(" " + substring, add_special_tokens=False)
            return self.target_tokens.index(tokens[0]), tokens

    @property
    def n_layers(self):
        return self.n_layers_target

    @property
    def n_layers_source(self):
        return len(getattr(getattr(self.source_model, self.MODEL_TARGET), self.LAYER_TARGET))

    @property
    def n_layers_target(self):
        return len(getattr(getattr(self.target_model, self.MODEL_TARGET), self.LAYER_TARGET))

    def compute_precision_at_1(self, estimated_probs, true_token_index):
        """
        Compute Precision@1 metric. From the outputs of the target (patched) model
        (estimated_probs) against the output of the source model, aka the 'true' token.
        Args:
        - estimated_probs: The estimated probabilities for each token as a torch.Tensor.
        - true_token_index: The index of the true token in the vocabulary.
        Returns:
        - precision_at_1: Precision@1 metric result.

        This is the evaluation method of the token identity from patchscopes: https://arxiv.org/abs/2401.06102
        Its used for running an evaluation over large datasets.
        """
        predicted_token_index = torch.argmax(estimated_probs)
        precision_at_1 = 1 if predicted_token_index == true_token_index else 0
        return precision_at_1

    def compute_surprisal(self, estimated_probs, true_token_index):
        """
        Compute Surprisal metric. From the outputs of the target (patched) model
        (estimated_probs) against the output of the source model, aka the 'true' token.
        Args:
        - estimated_probs: The estimated probabilities for each token as a torch.Tensor.
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
