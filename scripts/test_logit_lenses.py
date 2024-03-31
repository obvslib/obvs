import json
# Construct the path to the file
file_path = '../data/processed/sentences.json'

# Open and read the file
with open(file_path, 'r') as file:
    data = json.load(file)

from tqdm  import tqdm
from obvs.lenses import ClassicLogitLens, PatchscopeLogitLens
import random
from obvs.patchscope import ModelLoader
from transformers import AutoConfig
import numpy as np
device = 'gpu'
model_name = 'gpt2'
model = ModelLoader.load(model_name, device)
print (model.tokenizer)
# Two options - redo all of the preprocessing and follow the paper method exactly
# Or, skip it, shortcut to surprisal and precision calculations per sentence, see if the plot looks sensible.
# Option 2, then go back and do 1?
config = AutoConfig.from_pretrained('gpt2')
num_layers = config.num_hidden_layers
layers = list(range(0, num_layers))
test_amount = 5
surprisals_log_classic_ll = np.zeros((test_amount, num_layers))
surprisals_log_patchscope_ll = np.zeros((test_amount, num_layers))

import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings
# warnings.filterwarnings('default')  # Restore default warning behavior


counter = 0
for sentence, split in tqdm(data[0:test_amount]):
    # print (sentence)
    # print (model.tokenizer.encode(sentence))
    # print (type(sentence))
    tokenized_sentence = model.tokenizer.encode(sentence)
    rand_position = random.randrange(0, len(tokenized_sentence)-2)
    # truncated_tokenized_sentence = tokenized_sentence[:rand_position]

    ClassicLL = ClassicLogitLens(model_name, sentence, 'auto')
    PatchscopeLL = PatchscopeLogitLens(model_name, sentence, 'auto')

    # print (model.tokenizer.decode(truncated_tokenized_sentence))
    # ClassicLL.run(model.tokenizer.decode(truncated_tokenized_sentence), layers)
    ClassicLL.run(sentence, layers)
    PatchscopeLL.run(sentence, layers)


    # Prep for surprisal calc
    ClassicLL.target_token_id = tokenized_sentence[rand_position+1]
    ClassicLL.target_token_position = rand_position
    PatchscopeLL.target_token_id = tokenized_sentence[rand_position+1]
    PatchscopeLL.target_token_position = rand_position
    surprisals_across_layers_classic_LL = ClassicLL.compute_surprisal_at_position()
    surprisals_across_layers_patchscope_LL = PatchscopeLL.compute_surprisal_at_position()

    # print (surprisals_across_layers)
    surprisals_log_classic_ll[counter, :] = surprisals_across_layers_classic_LL
    surprisals_log_patchscope_ll[counter, :] = surprisals_across_layers_patchscope_LL
    counter += 1

    # Next: calculate surprisal and prec_1 for this experiment, output a plot as in Fig. 2
    # Just use a smaller set of sentences for now, to test the code works
    # Get final experiment into a script, and run .py file, instead of notebook