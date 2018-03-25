from collections import OrderedDict
import sys
from itertools import izip

gold_ent_name_file = sys.argv[1]
pred_ent_name_file = sys.argv[2]
#pred_file = sys.argv[3]
n_recall_sum = 0
count = 0
n_prec_sum = 0

def replace_kb_ent_in_resp(prob_memory_entities, pred_op):
        kb_name_list_unique = list(OrderedDict.fromkeys(prob_memory_entities))[:20]
        k = 0
        max_k = 20
        length = len(pred_op)
        replace_kb = True
        top_k_ent = []
        for j in range(len(pred_op)):
                if pred_op[j] in ['<pad_kb>','<nkb>','<unk>','</s>','<pad>']:
                        pred_op[j] = ''
                if pred_op[j]=='</e>':
                        length = j
                if pred_op[j].startswith('<kb>'):
                    if not replace_kb:
                        pred_op[j] = ''
                        continue
                    if k == len(kb_name_list_unique) or k == max_k:
                        replace_kb = False
                        pred_op[j] = ''
                        continue
                    top_k_ent.append(kb_name_list_unique[k])
                    k = k+1
        return top_k_ent

with open(gold_ent_name_file) as gold_lines, open(pred_ent_name_file) as pred_lines:#, open(pred_file) as sent_lines:
	#for gold_line, ent, pred in izip(gold_lines,pred_lines, sent_lines):
	for gold_line, ent in izip(gold_lines, pred_lines):
		#word_list = pred.strip().split(' ')
		#print 'sent', pred
                kb_count = 1
                #for word in pred.strip().split(' '):
                #        if word=='<kb>':
                #                 word_list.append('<kb>_'+str(kb_count))
                #                 kb_count = kb_count+1
                #        else:
                #                 word_list.append(word)
                #word_list = list(OrderedDict.fromkeys(word_list))
                if '|' in ent:
                        ent = ent.strip().split('|')
                else:
                        ent = [x.strip() for x in ent.strip().split(',')]
                top_k_ent = ent[:5]#replace_kb_ent_in_resp(ent, word_list)
		if '|' in gold_line:
			gold_entities = gold_line.rstrip().split('|')
		else:
			gold_entities = [x.strip() for x in gold_line.rstrip().split(',')]
		#print 'topk ', top_k_ent, '    ::::::  gold', gold_entities
		if len(gold_entities) > 0:
			pred_entities = top_k_ent
			#print 'len of pred entities', len(pred_entities)	
			n_topK = len(set(gold_entities).intersection(set(pred_entities)))
			n_recall_sum += n_topK*1.0/float(len(gold_entities))
			count += 1
			if len(pred_entities) > 0:
				n_prec_sum += n_topK*1.0/float(len(pred_entities))

avg_recall = n_recall_sum*100.0/float(count)
print gold_ent_name_file
print 'Avg. recall over= %f' % avg_recall
avg_prec= n_prec_sum*100.0/float(count)
print 'Avg. precision over= %f' % avg_prec
