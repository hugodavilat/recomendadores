#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import pandas as pd
import cornac

from cornac.eval_methods import RatioSplit
from cornac.models import MF, PMF, BPR, SVD, NMF, UserKNN, ItemKNN
from cornac.metrics import NDCG

class DependentsFeaturesCalculator():
    # Constructor -> Precisa de uma uir que seja um array de tuplas (user, item, rating).
    def __init__(self, uir):
        self.uir = uir
    
    def get_ndcgs(self):
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
        metrics = [NDCG(k=100)]

        # put it together in an experiment, voilà!
        experiment = cornac.Experiment(eval_method=rs, models=models, metrics=metrics, user_based=True, save_dir="/tmp/cornac")
        experiment.run()

        dict_ndcg_100 = {}

        for result in experiment.result:
            dict_ndcg_100[result.model_name] = result.metric_avg_results['NDCG@100']
        
        return dict_ndcg_100


# urm = pickle.load(open("../datasets/ml-100k/URM/001_URM.pk", 'rb'))
# df = pd.read_csv("../datasets/ml-100k/UIR/001_UIR.csv")
# uir = [tuple(row) for row in df.values]
# d = DependentsFeaturesCalculator(uir)
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

# # ItemKNN methods
# item_knn_cosine = cornac.models.ItemKNN(k=K, similarity="cosine", name="ItemKNN-Cosine")
# item_knn_pearson = cornac.models.ItemKNN(k=K, similarity="pearson", name="ItemKNN-Pearson")
# item_knn_adjusted = cornac.models.ItemKNN(k=K, similarity="cosine", mean_centered=True, name="ItemKNN-AdjustedCosine")