RECOMENDADOR = 'lfbca'
DATASET = 'gow'

result = '../' + RECOMENDADOR + '/rec_' + DATASET + '.out'
previsibility = '../previsibilidade/prev/' + DATASET + '_pred.txt'


class Merged:
    def __init__(self, id):
        self.id = id
        self.prec10 = 0
        self.prec50 = 0
        self.prec100 = 0
        self.rec = 0
        self.pred = 0

user_to_merged = {}

# linha 11: 0 11589 pre@10: 0.0 rec@10: 0.0 prec@50: 0.0 prec@100 0.0
with open(result) as result_file:
    lines = result_file.readlines()
    for line in lines[10:]:
        try:
            _, uid, _, prec10, _, rec, _, prec_50, _, prec_100  = line.strip().split(' ')
            uid, prec10, rec, prec_50, prec_100 = int(uid), float(prec10), float(rec), float(prec_50), float(prec_100)
            user_to_merged[uid] = Merged(uid)
            user_to_merged[uid].prec10, user_to_merged[uid].rec = prec10, rec
            user_to_merged[uid].prec50, user_to_merged[uid].prec100 = prec_50, prec_100
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

print('user_id\tprec@10\tprec@50\tprec@100\trec@10\tuser_predictability')

keys = [k for k in user_to_merged.keys()]
keys.sort()

for uid in keys:
    merged = user_to_merged[uid]
    print("{}\t{}\t{}\t{}\t{}\t{}".format(merged.id, merged.prec10, merged.prec50, merged.prec100, merged.rec, merged.pred))

    

