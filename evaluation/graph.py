from datasets import load_dataset
import matplotlib
import matplotlib.pyplot as plt

base_path = "evaluation/results/results"
model_name = "google/gemma-3-1b-it"
filename = "results_2025-11-15T22-41-45.271369.json"

json_file = "/".join([base_path, model_name, filename])

results_json = load_dataset("json", data_files=json_file, split="train")

benchmark_labels = {
    "arabic_mmlu:_average|0": "Arabic MMLU",
    "arabic_exams|0": "Arabic EXAMS",
    "acva:_average|0": "AVCA",
    "aratrust:_average|0": "AraTrust",
    "madinah_qa:_average|0": "MadinahQA",
}

tasks = dict.fromkeys(
    [
        "arabic_mmlu:_average|0",
        "arabic_exams|0",
        "acva:_average|0",
        "aratrust:_average|0",
        "madinah_qa:_average|0",
    ]
)

for task in tasks:
    tasks[task] = float(next(iter(results_json["results"][task][0].values()))) * 100
    print(f"{task:<25} {tasks[task]}")


# Plotting bar chart
matplotlib.use("agg")
plt.style.use("grayscale")

fig = plt.figure()
num_bars = range(len(tasks))
scores = list(tasks.values())
xtick_labels = [benchmark_labels[k] for k in tasks.keys()]

plt.bar(num_bars, scores, align="center", width=0.3)
plt.xticks(num_bars, xtick_labels, ha="center", rotation=0)
plt.ylim(0, 100)
# Source - https://stackoverflow.com/a/53073502
# Posted by Maroca, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-16, License - CC BY-SA 4.0

for i, score in enumerate(scores):
    plt.text(i, score + 1, f"{score:.2f}", ha="center")


plt.title("Arabic Benchmark Scores", fontweight="bold")
plt.xlabel("Benchmark", fontweight="bold")
plt.ylabel("Score", fontweight="bold")
plt.tight_layout()

model_name = model_name.replace(r"/", "__")
plt.savefig(f"evaluation/figures/{model_name}.png", dpi=fig.dpi)
