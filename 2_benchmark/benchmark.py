import os
import pandas as pd

from .models.qa_squad import qa_squad
from .models.llm import llm


import torch
print("cuda" if torch.cuda.is_available() else "cpu")

algorithms = {
    "big_bird": {
        "Model": lambda f: qa_squad(f, "what is the name of the proposed model ?", "FredNajjar/bigbird-QA-squad_v2.3"),
        "Parameters": lambda f: qa_squad(f, "what is the number of parameters of the proposed model ?", "FredNajjar/bigbird-QA-squad_v2.3"),
        "Training hardware": lambda f: qa_squad(f, "what is the hardware on which the proposed model was trained ?", "FredNajjar/bigbird-QA-squad_v2.3"),
    },
    "llm": {
        "Model": lambda f: llm(f, "what is the name of the proposed model ? (respond with the name only)"),
        "Parameters": lambda f: llm(f, "what is the number of parameters of the proposed model ? (respond with the number only)"),
        "Training hardware": lambda f: llm(f, "what is the hardware on which the proposed model was trained ? (respond with the hardware only)"),
    }
}



df = pd.read_csv("data/tables/train.csv")
os.makedirs("2_benchmark/results", exist_ok=True)



num_rows = len(df)

results = {}
for algorithm, algo_keys in algorithms.items():
    results[algorithm] = {}
    for key in algo_keys.keys():
        results[algorithm][key] = {
            "true": [""] * num_rows,
            "predicted": [""] * num_rows,
            "other": [""] * num_rows,
        }

for i in range(num_rows):
    row = df.iloc[i]
    for algorithm, algo_keys in algorithms.items():
        for key, func in algo_keys.items():
            results[algorithm][key]["true"][i] = row.get(key, "")
            if os.path.exists(f"data/files/train/{i}.txt"):
                with open(f"data/files/train/{i}.txt", "r") as f:
                    output = func(f.read())
                predicted = ""
                other = ""
                if isinstance(output, (list, tuple)):
                    if len(output) > 0:
                        predicted = output[0]
                    if len(output) > 1:
                        other = output[1]
                else:
                    predicted = output
                results[algorithm][key]["predicted"][i] = predicted
                results[algorithm][key]["other"][i] = other
            else:
                print(f"File {i}.txt does not exist")
                results[algorithm][key]["predicted"][i] = ""
                results[algorithm][key]["other"][i] = ""

            res_per_algo_key = pd.DataFrame(results[algorithm][key])
            res_per_algo_key.to_csv(f"2_benchmark/results/{algorithm}_{key}.csv", index=False)

        res_per_algo = {}
        for algo_key, data in results[algorithm].items():
            res_per_algo[f"{algo_key}_true"] = data["true"]
            res_per_algo[f"{algo_key}_predicted"] = data["predicted"]

        res_per_algo = pd.DataFrame(res_per_algo)
        res_per_algo.to_csv(f"2_benchmark/results/{algorithm}.csv", index=False)
