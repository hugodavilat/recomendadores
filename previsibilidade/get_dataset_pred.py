from prev import Explorability

DATASET = 'Gowalla'

data = '../data/' + DATASET + '_train.txt'
# res = 'prev/' + DATASET + '_pred.txt'

user_to_object = {}
ans = ""
train_data = open(data, 'r').readlines()

for eachline in train_data:
    uid, lid, freq = eachline.strip().split()
    uid, lid, freq = uid, lid, int(freq)
    if uid in user_to_object:
        user_to_object[uid].add_visit(lid, freq)
    else:
        user_to_object[uid] = Explorability(uid)
        user_to_object[uid].add_visit(lid, freq)

for user in user_to_object.keys():
    pred = user_to_object[user].get_max_pred()
    # ans += f'{user}, {pred}\n'
    print(f'{user}\t{pred}')

# with open(res, 'w') as ans_file:
#     ans_file.write(ans)
