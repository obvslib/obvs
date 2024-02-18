"""
run_benchmarks.py

Run the nnsight, pytorch & pytorch_compile benchmarks on a number of prompts
"""

import csv
from pathlib import Path
from nnsight_benchmark import run_benchmark as run_nnsight_benchmark
from pytorch_benchmark import run_benchmark as run_pytorch_benchmark
from typing import List, Tuple


def run_benchmark_on_prompt(prompt: str, model_id: str) -> List[Tuple]:
    """ Run the nnsight, pytorch & pytorch compile benchmarks on the given prompt.
        If write_to_csv is True, write the results to a .csv file"""

    results = []

    nnsight_runtime, nnsight_output = run_nnsight_benchmark(prompt, model_id)
    pytorch_runtime, pytorch_output = run_pytorch_benchmark(prompt, model_id)
    pytorch_compile_runtime, pytorch_compile_output = run_pytorch_benchmark(prompt, model_id,
                                                                            compile_model=True)

    results.append(('nnsight', nnsight_runtime, nnsight_output))
    results.append(('pytorch', pytorch_runtime, pytorch_output))
    results.append(('pytorch_compile', pytorch_compile_runtime, pytorch_compile_output))

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
                    csv_writer.writerow(['method', 'model_id', 'prompt', 'avg_runtime',
                                         'generated_text'])

            with open(csv_file_name, 'a') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([prompt_results[0], model_id, prompt, prompt_results[1],
                                     prompt_results[2]])


if __name__ == '__main__':
    run_benchmark_on_prompt('The quick brown fox jumps over the lazy', 'nickypro/tinyllama-110M')
