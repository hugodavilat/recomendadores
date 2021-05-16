class Merged:
    def __init__(self, id):
        self.id = id
        self.prec10 = 0
        self.prec50 = 0
        self.prec100 = 0
        self.acc10 = 0
        self.acc50 = 0
        self.acc100 = 0
        self.rec = 0
        self.pred = 0

user_to_merged = {}

def get_acc(top100_file, test_file):
    user_to_tops = {}
    count_tests = {}
    with open(top100_file) as t1f:
        for line in t1f.readlines():
            _, uid, top100 = line.strip().split()
            uid, top100 = int(uid), [int(x) for x in top100.split(',')]
            user_to_tops[uid] = (set(top100[:10]), set(top100[:50]), set(top100))
    with open(test_file) as tf:
        for line in tf.readlines():
            uid, lid, freq = line.strip().split()
            uid, lid, freq = int(uid), int(lid), int(freq)
            if uid not in count_tests:
                count_tests[uid] = [[0, 0], [0, 0], [0, 0]]
            # top 10
            try:
                if lid in user_to_tops[uid][0]:
                    count_tests[uid][0][0] += freq
                else:
                    count_tests[uid][0][1] += freq
                # top 50
                if lid in user_to_tops[uid][1]:
                    count_tests[uid][1][0] += freq
                else:
                    count_tests[uid][1][1] += freq
                # top 100
                if lid in user_to_tops[uid][2]:
                    count_tests[uid][2][0] += freq
                else:
                    count_tests[uid][2][1] += freq
            except Exception:
                pass
    for uid in count_tests.keys():
        try:
            user_to_merged[uid].acc10 = count_tests[uid][0][0]/(count_tests[uid][0][0] + count_tests[uid][0][1])
            user_to_merged[uid].acc50 = count_tests[uid][1][0]/(count_tests[uid][1][0] + count_tests[uid][1][1])
            user_to_merged[uid].acc100 = count_tests[uid][2][0]/(count_tests[uid][2][0] + count_tests[uid][2][1])
        except Exception:
            pass


get_acc('lfbca/result/gow_top_100.txt', 'data/Gowalla_test.txt')

# RECOMENDADOR = ('lfbca', 'lfbca')
# RECOMENDADOR = ('lore', 'LORE')
RECOMENDADOR = ('usg', 'USG')
# RECOMENDADOR = ('geosoca', 'GeoSoCa')
DATASET = ('gow', 'Gowalla')
# DATASET = ('yelp', 'Yelp')

result = RECOMENDADOR[1] + '/rec_' + DATASET[0] + '.out'
previsibility = 'previsibilidade/prev/' + DATASET[0] + '_pred.txt'
top100_file = RECOMENDADOR[1] + '/result/' + DATASET[0] + '_top_100.txt'
test_file = 'data/' + DATASET[1] + '_test.txt'

# precision
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
# accuracy
get_acc(top100_file, test_file)

# previsibility
with open(previsibility) as previsibility_file:
    for line in previsibility_file.readlines():
        try:
            uid, pred = line.strip().split('\t')
            user_to_merged[int(uid)].pred = float(pred)
        except Exception:
            pass

# print('user_id\tprec@10\tprec@50\tprec@100\trec@10\tuser_predictability')
out_str = 'user_id\tprec@10\tprec@50\tprec@100\tacc@10\tacc@50\tacc@100\trec@10\tuser_predictability\n'

keys = [k for k in user_to_merged.keys()]
keys.sort()

for uid in keys:
    merged = user_to_merged[uid]
    # print("{}\t{}\t{}\t{}\t{}\t{}".format(merged.id, merged.prec10, merged.prec50, merged.prec100, merged.rec, merged.pred))
    out_str += f'{merged.id}\t{merged.prec10}\t{merged.prec50}\t{merged.prec100}\t{merged.acc10}\t{merged.acc50}\t{merged.acc100}\t{merged.rec}\t{merged.pred}\n'

outfile = 'merge_results/result/' + RECOMENDADOR[0] + '_' + DATASET[0] + '_2.txt'
with open(outfile, 'w') as of:
    of.write(out_str)
