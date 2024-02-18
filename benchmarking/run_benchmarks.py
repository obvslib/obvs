"""
run_benchmarks.py

Run the nnsight, pytorch & pytorch_compile benchmarks on a number of prompts
"""

import csv
from pathlib import Path
from nnsight_benchmark import run_benchmark as run_nnsight_benchmark
from pytorch_benchmark import run_benchmark as run_pytorch_benchmark
from typing import List, Tuple
from functools import partial


def run_benchmark_on_prompt(prompt: str, model_id: str) -> List[Tuple]:
    """ Run the nnsight, pytorch & pytorch compile benchmarks on the given prompt.
        If write_to_csv is True, write the results to a .csv file"""

    results = []

    benchmarks_funcs = [('nnsight', run_nnsight_benchmark), ('pytorch', run_pytorch_benchmark),
                        ('pytorch_compile', partial(run_pytorch_benchmark, compile_model=True))]

    for benchmark_name, benchmark_func in benchmarks_funcs:
        avg_runtime, std_runtime, output = benchmark_func(prompt, model_id)
        results.append((benchmark_name, avg_runtime, std_runtime, output))

    return results


def run_benchmarks(prompts: List[str], model_id: str, write_to_csv: bool = True,
                   csv_file_name: str = 'python_benchmark_results.csv') -> None:
    """ Run benchmarks for all prompts with model_id. If write_to_csv == True, write the
        results to the given csv file"""

    for i, prompt in enumerate(prompts):
        prompt_results = run_benchmark_on_prompt(prompt, model_id)

        if write_to_csv:
            # check if csv file already exists
            if not Path(csv_file_name).exists():
                with open(csv_file_name, 'w') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(
                        ['method', 'model_id', 'prompt', 'avg_runtime', 'std_runtime',
                         'generated_text']
                    )

            with open(csv_file_name, 'a') as csv_file:
                csv_writer = csv.writer(csv_file)
                for result in prompt_results:
                    csv_writer.writerow([result[0], model_id, prompt, result[1], result[2]])


if __name__ == '__main__':
    benchmark_prompts = ['The quick brown fox jumps over the lazy',
                         'The capital of France is']

    run_benchmarks(benchmark_prompts, 'nickypro/tinyllama-110M', write_to_csv=True,
                   csv_file_name='python_benchmarks.csv')
