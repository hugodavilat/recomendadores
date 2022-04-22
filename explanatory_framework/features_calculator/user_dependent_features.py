#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import cornac
import json
import pandas as pd
import os

from cornac.eval_methods import RatioSplit
from cornac.models import MF, PMF, BPR, SVD, NMF, UserKNN, ItemKNN
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

class UserDependentsFeaturesCalculator():
    # Constructor -> Precisa de uma uir que seja um array de tuplas 
    # (user, item, rating).
    def __init__(self, train, test):
        self.train = train
        self.test = test
    
    def get_results(self):
        K = 50
        rs = RatioSplit(data=self.uir, test_size=0.2, seed=123)

        # initialize models, here we are comparing: Biased MF, PMF, and BPR
        models = [
            UserKNN(similarity="cosine", amplify=2.0, name="UserKNN-Amplified"),
            UserKNN(k=K, similarity="cosine", weighting="bm25", name="UserKNN-BM25"),
            UserKNN(k=K, similarity="cosine", name="UserKNN-Cosine"),
            UserKNN(k=K, similarity="cosine", weighting="idf", name="UserKNN-IDF"),
            ItemKNN(k=K, similarity="cosine", mean_centered=True, name="ItemKNN-AdjustedCosine"),
            BPR(seed=123),
            MF(seed=123),
            SVD(seed=123),
            PMF(seed=123),
            NMF(seed=123)
        ]

        # define metrics to evaluate the models
        metrics = [
            MAE(), RMSE(), Precision(k=100), Recall(k=100),
            NDCG(k=100), AUC(), MAP()
        ]

        # put it together in an experiment, voilà!
        experiment = cornac.Experiment(
            eval_method=rs, models=models,
            metrics=metrics, user_based=True, save_dir="/tmp/cornac"
        )
        experiment.run()
        result = {}
        for r in experiment.result:
            result[r.model_name] = r.metric_user_results # <- this is a dictionary
        return result

DS = "ml-100k"
# DS = "ml-1m"
# DS = "ml-10m"

METRICS = ["MAE", "RMSE", "AUC", "MAP", "NDCG@100", "Precision@100", "Recall@100"]
MODEL_NAMES = ["UserKNN-Amplified", "UserKNN-BM25", "UserKNN-Cosine", "UserKNN-IDF", "ItemKNN-AdjustedCosine", "BPR", "MF", "SVD", "PMF", "NMF"]
# ITEM_METRICS = pd.read_csv(f"../datasets/{DS}/item_metrics.csv")
USER_METRICS = pd.read_csv(f"../datasets/TS-split/{DS}/user_train_metrics.csv")
OUT = f"../datasets/TS-split/{DS}"

df = pd.read_csv(f"../datasets/TS-split/{DS}/uir.csv")
uir = [tuple(row) for row in df.values]

r = None
if os.path.isfile(f"{OUT}/result.json"):
    with open(f'{OUT}/result.json', 'r') as fp:
        r = json.load(fp)
else:
    d = UserDependentsFeaturesCalculator(uir)
    r = d.get_results()
    with open(f'{OUT}/result.json', 'w') as fp:
        json.dump(r, fp)

key_error_set_ = set()

for me in METRICS:
    out_str = "uid, " + ", ".join(MODEL_NAMES) + "\n"
    for uid in USER_METRICS['uid']:
        metrics = []
        for mo in MODEL_NAMES:
            try:
                metrics.append(f'{r[mo][me][str(uid-1)]:.5f}')
            except KeyError:
                key_error_set_.add(uid)
                metrics.append("0.0")
        out_str += str(uid) + ", " + ", ".join(metrics) + '\n'
    with open(f'{OUT}/{me}.csv', 'w') as fp:
        fp.write(out_str)

print(f'key_error_set = {key_error_set_}')