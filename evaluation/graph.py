import argparse
from datasets import load_dataset
import matplotlib
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        prog="graph",
        description="%(prog)s is a command-line interface used to quickly graph a model's benchmark results.",
    )

    parser.add_argument(
        "results_json",
        help="- Sprecify results JSON file (results*.json).",
    )

    args = parser.parse_args()
    print(args)

    if args.results_json:
        results_json = args.results_json

    graph(results_json)


def graph(results_json):
    results_json = load_dataset("json", data_files=results_json, split="train")
    model_name = results_json["config_general"]["model_name"][0]
    dtype = results_json["config_general"]["model_config"]["dtype"][0]
    print("Model:", model_name)
    print("Precision:", dtype)

    benchmark_labels = {
        "arabic_mmlu:_average|0": "Arabic MMLU",
        "arabic_exams|0": "Arabic EXAMS",
        "acva:_average|0": "ACVA",
        "aratrust:_average|0": "AraTrust",
        "madinah_qa:_average|0": "MadinahQA",
        "arc:challenge|0": "ARC (Challenge)",
        "hellaswag|0": "HellaSwag",
        "gsm8k|0": "GSM8K",
        "truthfulqa:mc|0": "TruthfulQA",
        "winogrande|0": "WinoGrande",
    }

    tasks = dict.fromkeys(benchmark_labels.keys())

    sum_total, sum_ar, sum_en = 0, 0, 0
    for i, task in enumerate(tasks):
        tasks[task] = float(next(iter(results_json["results"][task][0].values()))) * 100
        sum_total += tasks[task]
        if i < 5:
            sum_ar += tasks[task]
        else:
            sum_en += tasks[task]
        print(f"{task:<25} {tasks[task]}")

    avg_total = sum_total / 10
    avg_ar = sum_ar / 5
    avg_en = sum_en / 5

    # Plotting bar chart
    matplotlib.use("agg")
    plt.style.use("grayscale")

    fig_width = max(6, 1.2 * len(tasks))
    fig = plt.figure(figsize=(fig_width, 4))
    num_bars = range(len(tasks))
    scores = list(tasks.values())
    xtick_labels = [benchmark_labels[k] for k in tasks.keys()]

    plt.bar(num_bars, scores, align="center", width=0.3)
    plt.xticks(num_bars, xtick_labels, ha="center", rotation=0)
    plt.ylim(0, 100)

    for i, score in enumerate(scores):
        plt.text(i, score + 1, f"{score:.2f}", ha="center")

    plt.title(f"Benchmark Scores: {model_name} ({dtype})", fontweight="bold")
    plt.xlabel("Benchmark", fontweight="bold")
    plt.ylabel("Score", fontweight="bold")
    plt.tight_layout()

    avg_text = (
        f"Arabic Avg:  {avg_ar:.2f}\n"
        f"English Avg: {avg_en:.2f}\n"
        f"Total Avg:   {avg_total:.2f}"
    )

    plt.gca().text(
        0.01,
        0.96,
        avg_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    model_name = model_name.replace(r"/", "__")
    plt.savefig(f"evaluation/figures/{model_name}_{dtype}.png", dpi=fig.dpi)


if __name__ == "__main__":
    main()
