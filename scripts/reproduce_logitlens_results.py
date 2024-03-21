"""
Reproduce the results of the original logitlens blog post:
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens

(Note: There are two Colab notebooks associated with the blog post:
 The original notebook: https://colab.research.google.com/drive/1-nOE-Qyia3ElM17qrdoHAtGmLCPUZijg
 is based on TFv1 and does not work anymore, so we reproduce the results from the newer
 notebook: https://colab.research.google.com/drive/1MjdfK2srcerLrAJDRaJQKO0sUiZ-hQtA
)
"""

from obvs.lenses import ClassicLogitLens, PatchscopeLogitLens

prompt = """Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training
on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic
in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of
thousands of examples. By contrast, humans can generally perform a new language task from only
a few examples or from simple instructions – something which current NLP systems still largely
struggle to do. Here we show that scaling up language models greatly improves task-agnostic,
few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art 
finetuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion
parameters, 10x more than any previous non-sparse language model, and test its performance in
the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning,
with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3
achieves strong performance on many NLP datasets, including translation, question-answering, and
cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as
unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same
time, we also identify some datasets where GPT-3’s few-shot learning still struggles, as well as some
datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally,
we find that GPT-3 can generate samples of news articles which human evaluators have difficulty
distinguishing from articles written by humans. We discuss broader societal impacts of this finding
and of GPT-3 in general.""".replace("\n", " ")

substring = "Specifically, we train GPT-3, an autoregressive language model with 175 billion " \
            "parameters"

layers = list(range(0, 12))

# models: gpt2 125m, gpt2 1B, gpt-neo 125m
for model_name in ['gpt2', 'EleutherAI/gpt-neo-125M', 'gpt2-xl']:

    # run on both, classic and Patschcope logit lens
    for ll_type, ll_class in [('patchscope_logit_lens', PatchscopeLogitLens),
                              ('classic_logit_lens', ClassicLogitLens)]:
        ll = ll_class(model_name, prompt, 'auto')
        ll.run(substring, layers)
        fig = ll.visualize()
        fig.write_html(f'{model_name.replace("-", "_").replace("/", "_").lower()}_{ll_type}_logits_top_preds.html')
