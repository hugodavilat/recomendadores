import pandas as pd
import numpy as np

DS = "ml-1M"
METRICS = ["gini", "variation", "num_ratings", "rating_avg",
           "rating_std", "phi", "avg_phi", "entropy", "entropy_of_consumed_sum",
           "entropy_of_consumed_avg", "proportion_long_tailors"]

def get_feature_from_pandas(df, feature):
    return df[[feature]].to_numpy().flatten()

def normalize_feature(arr):
    arr_max, arr_min = max(arr), min(arr)
    return [(x-arr_min)/(arr_max-arr_min) for x in arr]

def get_out_str(feat_matrix):
    out_str = "uid, " + ", ".join(METRICS) + "\n"
    for usr in range(len(feat_matrix[0])):
        out_str += f'{usr+1}, ' + ", ".join([f'{feat_matrix[feat][usr]:.5f}' for feat in range(len(METRICS))]) + '\n'
    return out_str

df_features = pd.read_csv(f"../datasets/{DS}/user_metrics.csv",
                            sep='\s*,\s*', index_col=0, engine='python')

feat_matrix = []
for m in METRICS:
    feat_matrix.append(normalize_feature(get_feature_from_pandas(df_features, m)))

with open(f'../datasets/{DS}/normalized_user_metrics.csv', 'w') as norm_metrics_file:
    norm_metrics_file.write((get_out_str(feat_matrix)))
    

