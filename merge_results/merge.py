RECOMENDADOR = 'lfbca'
DATASET = 'Yelp'

result = '../' + RECOMENDADOR + '/rec_' + DATASET + '.out'
previsibility = '../previsibilidade/prev/' + DATASET + '_pred.txt'


class Merged:
    def __init__(self, id):
        self.id = id
        self.prec = 0
        self.rec = 0
        self.pred = 0

user_to_merged = {}

# linha 11: 0 11589 pre@10: 0.0 rec@10: 0.0
with open(result) as result_file:
    lines = result_file.readlines()
    for line in lines[10:]:
        try:
            _, uid, _, prec, _, rec  = line.strip().split(' ')
            uid, prec, rec = int(uid), float(prec), float(rec)
            user_to_merged[uid] = Merged(uid)
            user_to_merged[uid].prec, user_to_merged[uid].rec = prec, rec
        except Exception:
            pass

# line 0: 1	0.5724
with open(previsibility) as previsibility_file:
    for line in previsibility_file.readlines():
        try:
            uid, pred = line.strip().split('\t')
            user_to_merged[int(uid)].pred = float(pred)
        except Exception:
            pass

print('user_id\tprec@10\trec@10\tuser_predictability')

keys = [k for k in user_to_merged.keys()]
keys.sort()

for uid in keys:
    merged = user_to_merged[uid]
    print(f'{merged.id}\t{merged.prec}\t{merged.rec}\t{merged.pred}')

    

