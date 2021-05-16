from collections import defaultdict
from surprise import KNNBasic
from surprise import Dataset
from surprise.dataset import Reader, Trainset
from surprise.model_selection import PredefinedKFold

# Loading Training and Test data from train.csv and test.csv.
train_file = 'data/ml-100k/train.csv'
test_file = 'data/ml-100k/test.csv'
reader = Reader('ml-100k')
folds_files = [(train_file, test_file)]
data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()
trainset, testset = None, None
for _trainset, _testset in pkf.split(data):
    trainset = _trainset
    testset = _testset

# Train the algorithm on the trainset, and predict ratings for the testset.
sim_options = {'name': 'cosine',
               'user_based': True
               }
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)
predictions = algo.test(testset)

# Calculate error per user
idx = 0
usr_acc = []
err_max = -1
out_str_err = ""
out_str_acc = ""
for usr in range(943):
    usr_sum = 0
    for itm in range(10):
        pred = predictions[idx]
        usr_sum += abs(pred.est - pred.r_ui)
        idx += 1
    err = usr_sum/10
    err_max = max(err, err_max)
    usr_acc.append((usr+1, usr_sum/10))
    out_str_err += "{}\t{}\n".format(usr+1, usr_sum/10)

# Normalize errors -> ((err_max - err)/err_max)
for i in range(len(usr_acc)):
    acc = (err_max - usr_acc[i][1])/err_max
    usr_acc[i] = (usr_acc[i][0], acc)
    out_str_acc += "{}\t{}\n".format(usr_acc[i][0], acc)

# with open("data/digested/usr_vs_user_acc_ml-100k.csv", 'w') as acc_file:
#     acc_file.write(out_str_acc)