"""
utils.py

Utility functions that can be used by not only a Patchscope object
"""


def get_position_in_prompt(prompt, substring, tokenizer):
    """ Find the starting position of a substring in the prompt, and return the substring tokenized

    NB: The try: except block handles the difference between gpt2 and llama
    tokenization. Perhaps this can be better dealt with a seperate tokenizer
    class that handles the differences between the tokenizers. There are a
    few subtleties there, and tokenizing properly is important for getting
    the best out of your model.
    """
    if substring not in prompt:
        raise ValueError(f"{substring} not in {prompt}")
    try:
        tokens = tokenizer.encode(substring, add_special_tokens=False)
        return tokenizer.encode(prompt).index(tokens[0]), tokens
    except ValueError:
        tokens = tokenizer.encode(" " + substring, add_special_tokens=False)
        return tokenizer.encode(prompt).index(tokens[0]), tokens
