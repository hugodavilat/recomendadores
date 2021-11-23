#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import cornac
import pandas as pd

from cornac.eval_methods import RatioSplit
from cornac.models import MF, PMF, BPR, SVD, NMF, UserKNN, ItemKNN
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP

class UserDependentsFeaturesCalculator():
    # Constructor -> Precisa de uma uir que seja um array de tuplas (user, item, rating).
    def __init__(self, uir):
        self.uir = uir
    
    def get_results(self):
        K = 50

        rs = RatioSplit(data=self.uir, test_size=0.2, seed=123)

        # initialize models, here we are comparing: Biased MF, PMF, and BPR
        models = [
            UserKNN(similarity="cosine", amplify=2.0, name="UserKNN-Amplified"),
            # UserKNN(k=K, similarity="cosine", weighting="bm25", name="UserKNN-BM25"),
            # UserKNN(k=K, similarity="cosine", name="UserKNN-Cosine"),
            # UserKNN(k=K, similarity="cosine", weighting="idf", name="UserKNN-IDF"),
            # ItemKNN(k=K, similarity="cosine", mean_centered=True, name="ItemKNN-AdjustedCosine"),
            # BPR(seed=123),
            # MF(seed=123),
            # SVD(seed=123),
            # PMF(seed=123),
            # NMF(seed=123)
        ]

        # define metrics to evaluate the models
        metrics = [MAE(), RMSE(), Precision(k=100), Recall(k=100), NDCG(k=100), AUC(), MAP()]

        # put it together in an experiment, voilà!
        experiment = cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True, save_dir="/tmp/cornac")
        experiment.run()

        # for r in experiment.result:
        #     print(r.model_name)
        #     print(r.metric_user_results) # <- this is a dictionary

        return experiment.result


DS = "ml-100k"
# urm = pickle.load(open("../datasets/ml-100k/URM/001_URM.pk", 'rb'))
df = pd.read_csv(f"../datasets/{DS}/uir.csv")
uir = [tuple(row) for row in df.values]

d = UserDependentsFeaturesCalculator(uir)
r = d.get_results()

out_str = "uid,"


# ndcg_100 = d.get_ndcgs()
# print(ndcg_100)

# models used on the paper:
# UserKNN-Amplified, UserKNN-BM25, UserKNN-Cosine, UserKNN-IDF, ItemKNN-Adjusted, BPR, MF, SVD, PMF, and NMF.

# UserKNN methods
# user_knn_cosine = cornac.models.UserKNN(k=K, similarity="cosine", name="UserKNN-Cosine")
# user_knn_pearson = cornac.models.UserKNN(k=K, similarity="pearson", name="UserKNN-Pearson")
# user_knn_amp = cornac.models.UserKNN(k=K, similarity="cosine", amplify=2.0, name="UserKNN-Amplified")
# user_knn_idf = cornac.models.UserKNN(k=K, similarity="cosine", weighting="idf", name="UserKNN-IDF")
# user_knn_bm25 = cornac.models.UserKNN(k=K, similarity="cosine", weighting="bm25", name="UserKNN-BM25")

# exp = cornac.Experiment(
#   eval_method=rs,
#   models=[mf, pmf, bpr, wbpr, userknn, itemknn],
#   metrics=[mae, rmse, recall, ndcg, auc, mAP],
#   user_based=True
# )
# exp.run()

# for r in exp.result:
#   print(r.model_name)
#   user_results = r.metric_user_results # <- this is a dictionary
# That piece of code will help you to access the metric_user_results for each of the models in your experiment.