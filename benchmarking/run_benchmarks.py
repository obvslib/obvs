"""
run_benchmarks.py

Run the nnsight, pytorch & pytorch_compile benchmarks on a number of prompts
"""

import csv
from nnsight_benchmark import run_benchmark as run_nnsight_benchmark
from pytorch_benchmark import run_benchmark as run_pytorch_benchmark


def run_benchmark_on_prompt(prompt: str, model_id: str) -> dict:
    """ Run the nnsight, pytorch & pytorch compile benchmarks on the given prompt.
        If write_to_csv is True, write the results to a .csv file"""

    results = {}

    nnsight_runtime, nnsight_output = run_nnsight_benchmark(prompt, model_id)
    pytorch_runtime, pytorch_output = run_pytorch_benchmark(prompt, model_id)
    pytorch_compile_runtime, pytorch_compile_output = run_pytorch_benchmark(prompt, model_id,
                                                                            compile_model=True)

    results['nnsight'] = (nnsight_runtime, nnsight_output)
    results['pytorch'] = (pytorch_runtime, pytorch_output)
    results['pytorch_compile'] = (pytorch_compile_runtime, pytorch_compile_output)

    return results


if __name__ == '__main__':
    run_benchmark_on_prompt('The quick brown fox jumps over the lazy', 'nickypro/tinyllama-110M')
