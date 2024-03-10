"""
run_benchmarks.py

Run the nnsight, pytorch & pytorch_compile benchmarks on a number of prompts
"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import plotly.graph_objs as go
from nnsight_benchmark import run_benchmark as run_nnsight_benchmark
from pytorch_benchmark import run_benchmark as run_pytorch_benchmark


def run_benchmark_on_prompt(prompt: str, model_id: str) -> list[tuple]:
    """Run the nnsight, pytorch & pytorch compile benchmarks on the given prompt.
    If write_to_csv is True, write the results to a .csv file"""

    results = []

    benchmarks_funcs = [
        ("nnsight", run_nnsight_benchmark),
        ("pytorch", run_pytorch_benchmark),
        ("pytorch_compile", partial(run_pytorch_benchmark, compile_model=True)),
    ]

    for benchmark_name, benchmark_func in benchmarks_funcs:
        avg_runtime, std_runtime, output = benchmark_func(prompt, model_id)
        results.append((benchmark_name, avg_runtime, std_runtime, output))

    return results


def run_benchmarks(
    prompt: str,
    model_id: str,
    write_to_csv: bool = True,
    csv_file_name: str = "python_benchmark_results.csv",
) -> None:
    """Run benchmarks for all prompts with model_id. If write_to_csv == True, write the
    results to the given csv file"""

    prompt_results = run_benchmark_on_prompt(prompt, model_id)

    if write_to_csv:
        # check if csv file already exists
        if not Path(csv_file_name).exists():
            with open(csv_file_name, "w") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    [
                        "method",
                        "model_id",
                        "prompt",
                        "avg_runtime",
                        "std_runtime",
                        "generated_text",
                    ],
                )

        with open(csv_file_name, "a") as csv_file:
            csv_writer = csv.writer(csv_file)
            for result in prompt_results:
                csv_writer.writerow([result[0], model_id, prompt, result[1], result[2], result[3]])


def read_results_file(results_file: str) -> dict:
    """Read the results file and return its contents as a dictionary"""

    results = {}

    if not Path(results_file).exists():
        return

    # read results file
    with open(results_file) as csv_file:
        csv_reader = csv.reader(csv_file)
        # skip header
        next(csv_reader)
        for row in csv_reader:
            results[row[0]] = (row[1], row[2], row[3], row[4])

    return results


def plot_results(results: dict):
    """Plot the results of the benchmarks"""

    bar_names = []
    runtimes_avg = []
    runtimes_std = []
    colors = ["blue", "green", "red", "yellow"]

    for method, result_tuple in results.items():
        # get average runtime and std deviations
        bar_names.append(method)
        runtimes_avg.append(float(result_tuple[2]))
        runtimes_std.append(float(result_tuple[3]))
        model_name = result_tuple[0]
        prompt = result_tuple[1]

    fig = go.Figure()

    print(bar_names, runtimes_avg, runtimes_std)

    # Add bar trace
    fig.add_trace(
        go.Bar(
            x=bar_names,
            y=runtimes_avg,
            error_y=dict(type="data", array=runtimes_std, visible=True),
            marker=dict(color=colors),
        ),
    )

    # Update layout
    fig.update_layout(
        title=f'Text Generation Benchmark. Model: "{model_name}", n_new_tokens: 20, Prompt: "{prompt}"',
        xaxis=dict(title="Method"),
        yaxis=dict(title="Avg runtimes"),
    )

    fig.show()


if __name__ == "__main__":
    benchmark_prompt = "The quick brown fox jumps over the lazy"

    run_benchmarks(
        benchmark_prompt,
        "nickypro/tinyllama-110M",
        write_to_csv=True,
        csv_file_name="python_benchmarks.csv",
    )
