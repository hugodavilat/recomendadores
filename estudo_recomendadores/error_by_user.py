from collections import defaultdict
# from surprise.prediction_algorithms.matrix_factorization import SVDpp
from surprise.prediction_algorithms.knns import KNNWithZScore
from surprise import Dataset
from surprise.dataset import Reader
from surprise.model_selection import PredefinedKFold

# Loading Training and Test data from train.csv and test.csv.
train_file = '../data/ml-100k/train.csv'
test_file = '../data/ml-100k/test.csv'
reader = Reader('ml-100k')
folds_files = [(train_file, test_file)]
data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()
trainset, testset = None, None
for _trainset, _testset in pkf.split(data):
    trainset = _trainset
    testset = _testset

# Train the algorithm on the trainset, and predict ratings for the testset.
# sim_options = {'name': 'cosine',
#                'user_based': False
#                }
print("Fitting Trainset...")
algo = KNNWithZScore()
algo.fit(trainset)

print("Predicting for Testset...")
# Prediction(uid='943', iid='215', r_ui=5.0, est=3.4267981399690726,
#            details={'actual_k': 40, 'was_impossible': False})
predictions = algo.test(testset)

# Error per tuple <user,item>
# out_str_usr_itm = "user,item,rating,est,err\n"
out_str_usr_itm = ""
for pred in predictions:
    out_str_usr_itm += f'{pred.uid},{pred.iid},{pred.r_ui},{pred.est},{abs(pred.r_ui-pred.est)}\n'

# Average error per user
idx = 0
# out_str_usr_avg = "user,err_avg\n"
out_str_usr_avg = ""
for usr in range(943):
    usr_sum = 0
    for itm in range(10):
        pred = predictions[idx]
        usr_sum += abs(pred.est - pred.r_ui)
        idx += 1
    avg_err = usr_sum/10
    out_str_usr_avg += f'{usr+1},{avg_err}\n'

# csv1 = out_str_usr_itm
# csv2 = out_str_usr_avg
# print(out_str_usr_itm)
# print(out_str_usr_avg)
with open("out/ml-100k/user_item_abs_err_KNNWithZScore.csv", 'w') as csv1:
    csv1.write(out_str_usr_itm)
with open("out/ml-100k/user_err_avg_KNNWithZScore.csv", 'w') as csv2:
    csv2.write(out_str_usr_avg)
