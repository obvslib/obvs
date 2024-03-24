"""

Use activation patching to study indirect object identification (IOI) on gpt2-small

The task in IOI is to identify the correct indirect object in a sentence like:
    "After John and Mary went to the store, Mary gave a bottle of milk to"

With activation patching, we can analyze which parts of a model contribute to the task.

Resources:
Interpretability in the wild paper: https://arxiv.org/pdf/2211.00593.pdf
Neel Nandas Activation patching with TL Colab:
    https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Activation_Patching_in_TL_Demo.ipynb

In this script, we implement activation patching via the patchscope framework.

"""

from obvs.patchscope import SourceContext, TargetContext, Patchscope
from obvs.vis import create_annotated_heatmap


# define metric
def ioi_metric(patched_logits, clean_logits, corrupt_logits, correct_idx, incorrect_idx):
    """ Metric for checking correctness of indirect object identification
        the normalized_logit_diff is constructed so that it is 1, if the output logits are the
        same as in the clean run, and 0 if the output logits are the same as in the corrupted run
        it increases, if the patched activation contribute in making the output logits
        more like in the clean run
    """

    # get the difference in logits of the correct token and the incorrect token at the last
    # position
    patched_logit_diff = (patched_logits[-1, correct_idx] - patched_logits[-1, incorrect_idx])
    clean_logit_diff = (clean_logits[-1, correct_idx] - clean_logits[-1, incorrect_idx])
    corrupt_logit_diff = (corrupt_logits[-1, correct_idx] - corrupt_logits[-1, incorrect_idx])

    return (patched_logit_diff - corrupt_logit_diff) / (clean_logit_diff - corrupt_logit_diff)

# setup
# the clean prompt produces our baseline answer, the corrupted prompt will be patched with
# activations from the clean prompt run later on
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = "After John and Mary went to the store, John gave a bottle of milk to"

# setup patchscope
source_context = SourceContext(prompt=clean_prompt, model_name='gpt2')
target_context = TargetContext(prompt=corrupted_prompt, model_name='gpt2', max_new_tokens=1)
patchscope = Patchscope(source_context, target_context)


# get the indices of the correct and incorrect tokens
correct_index = patchscope.tokenizer(" John")["input_ids"][0]
incorrect_index = patchscope.tokenizer(" Mary")["input_ids"][0]


# first, do a forward pass on the corrupted prompt to get the corrupted logits
patchscope.source.prompt = corrupted_prompt
patchscope.source_forward_pass()
corrupted_logits = patchscope.source_output.detach()


# then, do a forward pass on the clean prompt to get the clean logits
patchscope.source.prompt = clean_prompt
patchscope.source_forward_pass()
clean_logits = patchscope.source_output.detach()


# now, do activation patching
metrics = []
n_layers = 3

# loop over all layers of interest
for layer in range(n_layers):

    layer_metrics = []

    # loop over all token positions
    for pos in range(len(patchscope.source_tokens)):

        # set the layer and position for patching
        patchscope.source.layer = layer
        patchscope.target.layer = layer
        patchscope.source.position = pos
        patchscope.target.position = pos

        # run the patchscope
        patchscope.run()

        # get the patched logits and calculate the logit difference
        patched_logits = patchscope.logits()
        layer_metrics.append(ioi_metric(patched_logits, clean_logits, corrupted_logits,
                                        correct_index, incorrect_index).item())

    metrics.append(layer_metrics)

# create x and y ticks for the heatmap


fig = create_annotated_heatmap(metrics, None, patchscope.source_words, list(range(n_layers)))
fig.show()


