import math
import json
import re
import numpy as np

from collections import defaultdict

DATASET = 'Yelp'
TRAIN_DATA_RATIO = 0.75
NUM_WINDOWS = 2

checkins = '../data/' + DATASET + '_checkins.txt'
data_json = '../data/' + DATASET + '_' + str(NUM_WINDOWS) + '_windows.json'
user_to_checkins = defaultdict(list)
user_data = {}

# terrible way of doing this. No other is possible
def fix_json_indent(text):
    return re.sub(r'\{', '{\n', re.sub(r'\]\,', '],\n', re.sub(r'\}\,', '\n},\n', text)))

def entropy_kontoyiannis(sequence):
    if not sequence:
        return 0.0    
    lambdas = 0
    n = len(sequence)
    for i in range(n):
        current_sequence = ''.join(sequence[0:i])
        match = True
        k = i
        while match and k < n:
            k += 1
            match = ''.join(sequence[i:k]) in current_sequence
        lambdas += (k - i)
    return (1.0 * len(sequence) / lambdas) * np.log2(len(sequence))


def max_predictability(S, N):
    if S == 0.0 or N <= 1:
        return 1.0
    for p in np.arange(0.0001, 1.0000, 0.0001):
        h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        pi_max = h + (1 - p) * math.log2(N - 1) - S
        if pi_max <= 0.001:
            return round(p, 5)
    return 0.0


# Example
# x = ['H', 'W', 'H', 'W', 'P', 'S', 'H', 'W', 'H', 'W', 'H', 'W', 'H', 'W']
# ent = entropy_kontoyiannis(x)
# pred = max_predictability(ent, len(set(x)))
# print(pred)

with open(checkins) as cf:
    for line in cf.readlines():
        uid, lid, ts = line.strip().split()
        uid, lid, ts = int(uid), int(lid), int(float(ts))
        user_to_checkins[uid].append((ts, lid))

for uid in user_to_checkins.keys():
    checkins = user_to_checkins[uid]
    checkins.sort()
    windows = []
    train = []
    test = []
    for i in range(NUM_WINDOWS):
        start = i*len(checkins)//NUM_WINDOWS
        end = (i+1)*len(checkins)//NUM_WINDOWS
        train_end = int((end-start)*TRAIN_DATA_RATIO)+start
        train, test = checkins[start:train_end], checkins[train_end:end]
        train_places = [x[1] for x in train]
        test_places = [x[1] for x in test]
        train_ts = [x[0] for x in train]
        test_ts = [x[0] for x in test]
        ent = entropy_kontoyiannis([str(t) for t in train_places])
        pred = max_predictability(ent, len(set(train_places)))
        user_data[str(uid) + '_' + str(i)] = {
            'train_lid': train_places,
            'test_lid': test_places,
            'train_ts': train_ts,
            'test_ts': test_ts,
            'pred': pred
        }

out_str = json.dumps(user_data)
out_str = fix_json_indent(out_str)

with open(data_json, 'w') as outfile:
    outfile.write(out_str)