#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import cornac
import json
import pandas as pd
import os

from cornac.eval_methods import RatioSplit, StratifiedSplit
from cornac.models import MF, PMF, BPR, SVD, NMF, UserKNN, ItemKNN
from cornac.metrics import MAE, MSE, RMSE, AUC, MAP, Precision, Recall, NDCG

class UserDependentsFeaturesCalculator():
    # Constructor -> Precisa de uma uir que seja um array de tuplas (user, item, rating).
    def __init__(self, uir):
        self.uir = uir
    
    def get_results(self):
        K = 50

        ss = StratifiedSplit(data=self.uir, test_size=0.2, seed=123, exclude_unknowns=False)

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
        metrics = [MAE(), MSE(), RMSE(), AUC(), MAP(), Precision(k=20), Recall(k=20), NDCG(k=20), Precision(k=100), Recall(k=100), NDCG(k=100)]

        # put it together in an experiment, voilà!
        experiment = cornac.Experiment(eval_method=ss, models=models, metrics=metrics, user_based=True, save_dir="/tmp/cornac/")
        experiment.run()

        result = {}
        for r in experiment.result:
            result[r.model_name] = r.metric_user_results # <- this is a dictionary

        return result

# DS = "ml-100k"
DS = "ml-1M"

METRICS = ["MAE", "MSE", "RMSE", "AUC", "MAP", "Precision@20", "Recall@20", "NDCG@20", "Precision@100", "Recall@100", "NDCG@100"]
MODEL_NAMES = ["UserKNN-Amplified", "UserKNN-BM25", "UserKNN-Cosine", "UserKNN-IDF", "ItemKNN-AdjustedCosine", "BPR", "MF", "SVD", "PMF", "NMF"]
USER_METRICS = pd.read_csv(f"../datasets/{DS}/user_metrics_v2.csv")
OUT = f"../user_features_table/{DS}"

df = pd.read_csv(f"../datasets/{DS}/uir.csv")
uir = []
for row in df.values:
    uir.append((str(int(row[0])), str(int(row[1])), float(row[2]), 1))
uir.sort()

r = None
if os.path.isfile(f"{OUT}/result_v2.json"):
    with open(f'{OUT}/result_v2.json', 'r') as fp:
        r = json.load(fp)
else:
    d = UserDependentsFeaturesCalculator(uir)
    r = d.get_results()
    with open(f'{OUT}/result_v2.json', 'w') as fp:
        json.dump(r, fp)

for me in METRICS:
    out_str = "uid, " + ", ".join(MODEL_NAMES) + "\n"
    for uid in USER_METRICS['uid']:
        metrics = []
        for mo in MODEL_NAMES:
            metrics.append(f'{r[mo][me][str(uid-1)]:.5f}')
        out_str += str(uid) + ", " + ", ".join(metrics) + '\n'
    with open(f'{OUT}/{me}.csv', 'w') as fp:
        fp.write(out_str)