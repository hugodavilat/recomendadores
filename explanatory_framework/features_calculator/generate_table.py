#!/Users/hugo/Documents/recomendadores/explanatory_framework/venv/bin/python

import pickle
import pandas as pd

from dependent_features import DependentsFeaturesCalculator
from independent_features import IndependentsFeaturesCalculator

# DATA = "ml-100k"
# DATA = "ml-1M"

for DATA in ["ml-1M"]:
    DS = f"../datasets/{DATA}"
    TABLES = "../features_table"

    out_str = "sample_number, space_size, shape, density, rpu, rpi, gini_u, "
    out_str += "gini_i, pop_avg, pop_std, pop_skew, pop_kurtosis, long_tail_avg, "
    out_str += "long_tail_std, long_tail_skew, long_tail_kurtosis, "
    out_str += "UserKNN-Amplified, UserKNN-BM25, UserKNN-Cosine, UserKNN-IDF, "
    out_str += "ItemKNN-AdjustedCosine, BPR, MF, SVD, PMF, NMF\n"

    for sample_number in range(1, 601):
        try:
            sample_number_str = "%03d" % sample_number
            sample_out_str = f'{sample_number_str}, '
            # open samples and create object
            urm = pickle.load(
                open(f"{DS}/URM/{sample_number_str}_URM.pk", 'rb'))
            independent_features = IndependentsFeaturesCalculator(urm)
            df = pd.read_csv(f"{DS}/UIR/{sample_number_str}_UIR.csv")
            uir = [tuple(row) for row in df.values]
            dependent_features = DependentsFeaturesCalculator(uir)

            # calculate independent features for sample
            sample_out_str += f'{independent_features.space_size()}, '
            sample_out_str += f'{independent_features.shape():.4f}, '
            sample_out_str += f'{independent_features.density():.4f}, '
            sample_out_str += f'{independent_features.rpu():.4f}, '
            sample_out_str += f'{independent_features.rpi():.4f}, '
            sample_out_str += f'{independent_features.gini_u():.4f}, '
            sample_out_str += f'{independent_features.gini_i():.4f}, '
            sample_out_str += f'{independent_features.pop_avg():.4f}, '
            sample_out_str += f'{independent_features.pop_std():.4f}, '
            sample_out_str += f'{independent_features.pop_skew():.4f}, '
            sample_out_str += f'{independent_features.pop_kurtosis():.4f}, '
            sample_out_str += f'{independent_features.long_tail_avg():.4f}, '
            sample_out_str += f'{independent_features.long_tail_std():.4f}, '
            sample_out_str += f'{independent_features.long_tail_skew():.4f}, '
            sample_out_str += f'{independent_features.long_tail_kurtosis():.4f}'

            # calculate dependent features for sample
            algos = [
                "UserKNN-Amplified", "UserKNN-BM25", "UserKNN-Cosine", "UserKNN-IDF", 
                "ItemKNN-AdjustedCosine", "BPR", "MF", "SVD", "PMF", "NMF"
            ]
            ndcgs = dependent_features.get_ndcgs()
            for algo in algos:
                sample_out_str += f', {ndcgs[algo]:.4f}'
            out_str += sample_out_str
            out_str += "\n"

            with open(f'{TABLES}/{DATA}/NDCG100-2.csv', 'w') as out_file:
                out_file.write(out_str)
        except Exception:
            print(f"Error on sample {sample_number_str}")

