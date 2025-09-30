import os
import pandas as pd

from .models.qa_squad import qa_squad


import torch
print("cuda" if torch.cuda.is_available() else "cpu")


#models = [lambda f: qa_squad(f, "what is the name of the proposed model ?")]


algorithms = {
    "big_bird": {
        "Model": lambda f: qa_squad(f, "what is the name of the proposed model ?", "FredNajjar/bigbird-QA-squad_v2.3"),
        "Parameters": lambda f: qa_squad(f, "what is the number of parameters of the proposed model ?", "FredNajjar/bigbird-QA-squad_v2.3"),
    }
}



df = pd.read_csv("data/tables/train.csv")
os.makedirs("2_benchmark/results", exist_ok=True)

#data = {
#    "Model_predicted": [],
#    "Model_true": [],
#    "other_predicted": []
#}


#keys = ["Model", "Parameters", "Training hardware"]

#data = {
#    "Model" : {
#        "true": [],
#        "predicted": [],
#        "other": []
#    },
#
#    "Parameters" : {
#        "true": [],
#        "predicted": [],
#        "other": []
#    },
#
#    "Training hardware" : {
#        "true": [],
#        "predicted": [],
#        "other": []
#    }
#}

#data = {key: {
#    "true": [],
#    "predicted": [],
#    "other": []
#} for key in keys}


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


            





###################################


#for i in range(1000):
#    if os.path.exists(f"data/files/train/{i}.txt"):
#        print(f"File {i}.txt exists")
#        with open(f"data/files/train/{i}.txt", "r") as f:
#            try: 
#                (best, other) = qa_squad(f.read(), "What is the name of the proposed model ?")
#            except Exception as e:
#                (best, other) = ("", "")
#                print(f"Error: {e}")
#                continue
#
#            
#            data["Model_predicted"].append(best)
#            data["other_predicted"].append(other)
#            data["Model_true"].append(df.iloc[i]["Model"])
#        #print(pd.DataFrame(data).to_string(max_rows=None, max_cols=None))
#    else:
#        print(f"File {i}.txt does not exist")
#
#    print(pd.DataFrame(data).to_string(max_rows=None, max_cols=None))
#
#    df_data = pd.DataFrame(data)
#
#    df_data.to_csv("2_benchmark/prediction.csv", index=False)