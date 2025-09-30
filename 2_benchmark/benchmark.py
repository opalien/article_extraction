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


results = {algorithm: {key: {
    "true": [],
    "predicted": [],
    "other": []
} for key in algorithms[algorithm].keys()} for algorithm in algorithms}

for i in range(len(df)):
    for algorithm in algorithms:
        for key in algorithms[algorithm].keys():
            results[algorithm][key]["true"].append(df.iloc[i][key])
            if os.path.exists(f"data/files/train/{i}.txt"):
                with open(f"data/files/train/{i}.txt", "r") as f:
                    output = algorithms[algorithm][key](f.read())
                    results[algorithm][key]["predicted"].append(output[0])
                    results[algorithm][key]["other"].append(output[1])
            else:
                results[algorithm][key]["predicted"].append("")
                results[algorithm][key]["other"].append("")
            
            res_per_algo_key = pd.DataFrame(results[algorithm][key])
            res_per_algo_key.to_csv(f"2_benchmark/results/{algorithm}_{key}.csv", index=False)

            #keys = [f"{key}_{suffix}" for key in keys for suffix in ("true", "predicted")]


            res_per_algo = {}
            for key in results[algorithm].keys():
                res_per_algo[key+"_true"] = results[algorithm][key]["true"]
                res_per_algo[key+"_predicted"] = results[algorithm][key]["predicted"]
            
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