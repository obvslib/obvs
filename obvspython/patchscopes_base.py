from typing import Optional
from copy import deepcopy
from abc import ABC, abstractmethod

import torch


class PatchscopesBase(ABC):
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
        return "model", "layers"

    @abstractmethod
    def source_forward_pass(self):
        pass

    @abstractmethod
    def map(self):
        pass

    @abstractmethod
    def target_forward_pass(self):
        pass

    @abstractmethod
    def run(self):
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

    def get_position(self, force=False):
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
        tokens.insert(0, ' ')
        tokens[:len(input_tokens)] = input_tokens
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
        Find the position of the target tokens in the source tokens
        """
        if substring not in self.source.prompt:
            raise ValueError(f"{substring} not in {self.source.prompt}")
        tokens = self.tokenizer.encode(substring)
        return self.source_tokens.index(tokens[0])

    def find_in_target(self, substring):
        """
        Find the position of the target tokens in the source tokens
        """
        if substring not in self.target.prompt:
            raise ValueError(f"{substring} not in {self.target.prompt}")
        tokens = self.tokenizer.encode(substring)
        return self.target_tokens.index(tokens[0])

    @property
    def n_layers(self):
        return len(getattr(getattr(self.target_model, self.MODEL_TARGET), self.LAYER_TARGET))

    def get_activation_pair(self, string_a: str, string_b: Optional[str] = None):
        """
        Get the activations for two strings for activation steering.
        :param string_a: The first string to compare
        :param string_b: The second string to compare
        :param bomb: If True, the string will be repeated to fill the source prompt
        :return: The activations for the two strings
        """
        tokens_a, tokens_b = self.justify(string_a, string_b)

        position = range(len(tokens_a))

        source = deepcopy(self.source)
        source.prompt = self.tokenizer.decode(tokens_a)
        source.position = position

        print(f"Getting representation with settings: {source}")
        activations_a = self.get_source_hidden_state(source)

        source.prompt = self.tokenizer.decode(tokens_b)
        print(f"Getting representation with settings: {source}")
        activations_b = self.get_source_hidden_state(source)

        return activations_a, activations_b

    def justify(self, string_a: str, string_b: Optional[str] = None):
        if not string_a.startswith(" "):
            string_a = " " + string_a
        tokens_a = self.tokenizer.encode(string_a, add_special_tokens=False)

        # If string_b is not provided, use spaces
        if string_b is None:
            string_b = " "
        elif not string_b.startswith(" "):
            string_b = " " + string_b
        if "lama" in self.source.model_name:
            string_b = " " + string_b

        tokens_b = self.tokenizer.encode(string_b, add_special_tokens=False)

        # Pad the shortest string with spaces
        if len(tokens_a) > len(tokens_b):
            tokens_b = tokens_b + self.tokenizer.encode(" ", add_special_tokens=False) * (len(tokens_a) - len(tokens_b))
        elif len(tokens_a) < len(tokens_b):
            tokens_a = tokens_a + self.tokenizer.encode(" ", add_special_tokens=False) * (len(tokens_b) - len(tokens_a))

        print(f"Activation pair created: tokens_a: {len(tokens_a)}, tokens_b: {len(tokens_b)}, source_tokens: {len(self.source_tokens)}")

        assert len(tokens_a) == len(tokens_b)
        assert len(tokens_a) <= len(self.source_tokens)

        return tokens_a, tokens_b

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
        # To avoid log(0) issues, add a small constant to the probabilities
        estimated_probs = torch.clamp(estimated_probs, min=1e-12)
        surprisal = -torch.log(estimated_probs[true_token_index])
        return surprisal.item()
