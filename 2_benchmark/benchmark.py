import os
import pandas as pd

from .models.qa_squad import qa_squad


import torch
print("cuda" if torch.cuda.is_available() else "cpu")


models = [lambda f: qa_squad(f, "what is the full name of the proposed machine learning model ? It is not in the references")]

df = pd.read_csv("data/tables/train.csv")

data = {
    "Model_predicted": [],
    "Model_true": []
}


#i = 0
#while os.path.exists(f"data/files/train/{i}.txt"):
#    with open(f"data/files/train/{i}.txt", "r") as f:
#        data["Model"].append(qa_squad(f.read(), "what is the full name of the proposed model ? It's not in the references"))
#    print(pd.DataFrame(data).to_string(max_rows=None, max_cols=None))
#    i += 1


for i in range(1000):
    if os.path.exists(f"data/files/train/{i}.txt"):
        print(f"File {i}.txt exists")
        with open(f"data/files/train/{i}.txt", "r") as f:
            data["Model_predicted"].append(qa_squad(f.read(), "what is the full name of the proposed model ? It's not in the references"))
            data["Model_true"].append(df.iloc[i]["Model"])
        #print(pd.DataFrame(data).to_string(max_rows=None, max_cols=None))
    else:
        print(f"File {i}.txt does not exist")

    print(pd.DataFrame(data).to_string(max_rows=None, max_cols=None))

    df_data = pd.DataFrame(data)

    df_data.to_csv("2_banchmark/prediction.csv", index=False)