import codecs, sys, pickle
from itertools import izip

# mem_target_file = sys.argv[1]
# gold_target_file = sys.argv[2]

# overlap_ratio_sum = 0
# n_lines = 0
# n_err = 0
# with open(mem_target_file) as f1, open(gold_target_file) as f2:
# 	for mem_target_line, gold_target_line in izip(f1, f2):
# 		mem_target_line = mem_target_line.rstrip()
# 		gold_target_line = gold_target_line.rstrip()

# 		if len(mem_target_line) > 0:
# 			mem_target_tokens = mem_target_line.split('|')
# 		else:
# 			mem_target_tokens = []

# 		if len(gold_target_line)>0:
# 			gold_target_tokens = gold_target_line.split('|')
# 		else:
# 			gold_target_tokens = []

# 		common_tokens = set(mem_target_tokens).intersection(set(gold_target_tokens))

# 		if len(gold_target_tokens)>0:
# 			overlap_ratio = len(common_tokens)*1.0/len(gold_target_tokens)
# 			overlap_ratio_sum += overlap_ratio
# 			n_lines += 1
# 			print 'overlap_ratio = %f' % overlap_ratio
# 		else:
# 			n_err += 1

# avg_overlap_ratio = overlap_ratio_sum*1.0/n_lines

# print 'Avg. overlap_ratio = %f' % avg_overlap_ratio
# print 'n_err = %d' % n_err

data_pkl_file = sys.argv[1]
# data_pkl_file = 'model_2/dump/train_data_file.pkl'

data = pickle.load(open(data_pkl_file,'r'))

overlap_ent = [set(data[i][2]).intersection(set(data[i][8])) for i in range(len(data))]
overlap_ent_filt = [(x - set([0])) for x in overlap_ent]
gold_target_len = [len(set(data[i][2]) - set([0])) for i in range(len(data))]

overlap_ratio = [len(x)*1.0/y for x,y in izip(overlap_ent_filt, gold_target_len) if y != 0]

avg_overlap_ratio = sum(overlap_ratio)*1.0/len(overlap_ratio)
avg_nonzero_overlap_ratio = len([x for x in overlap_ent_filt if len(x)>0])*1.0/len(data)

print 'Avg. overlap_ratio = %f' % avg_overlap_ratio
print 'avg_nonzero_overlap_ratio = %f' % avg_nonzero_overlap_ratio


