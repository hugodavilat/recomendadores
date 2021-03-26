from prev import Explorability

DATASET = 'Yelp'
DATASET = 'Gowalla'

# data = '../data/Yelp_check_ins.txt'
data = '../data/Gowalla_checkins.txt'

user_to_object = {}
ans = ""
train_data = open(data, 'r').readlines()

for eachline in train_data:
    # uid, lid, freq = eachline.strip().split()
    # uid, lid, freq = uid, lid, int(freq)
    uid, lid, _ = eachline.strip().split()
    if uid in user_to_object:
        # user_to_object[uid].add_visit(lid, freq)
        user_to_object[uid].add_visit_sg(lid)
    else:
        user_to_object[uid] = Explorability(uid)
        # user_to_object[uid].add_visit(lid, freq)
        user_to_object[uid].add_visit(lid)


for user in user_to_object.keys():
    pred = user_to_object[user].get_max_pred()
    print(f'{user}\t{pred}')
