import sys
from collections import OrderedDict
import re

decoder_dir = sys.argv[1]
kvmem_dir = sys.argv[2]
test_type = sys.argv[3]
state = sys.argv[4]
type = sys.argv[5]
pred_file = decoder_dir+'/test_output_'+test_type+'_'+state+'/pred_sent.txt'
ent_file = decoder_dir+'/test_output_'+test_type+'_'+state+'/top20_ent_from_'+type+'.txt'

def replace_kb_ent_in_resp(prob_memory_entities, pred_op):
        kb_name_list_unique = list(OrderedDict.fromkeys(prob_memory_entities))[:20]
        k = 0
        max_k = 10
        length = len(pred_op)
        replace_kb = True
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
                    pred_op[j] = kb_name_list_unique[k]
                    k = k+1
        pred_op = pred_op[:length]
        pred_op = re.sub(' +',' ',' '.join(pred_op)).strip()
        return pred_op

with open(pred_file) as pred_lines, open(ent_file) as ent_lines:
	for pred, ent in zip(pred_lines, ent_lines):
		word_list = pred.strip().split(' ')
		
		kb_count = 1
		for word in pred.strip().split(' '):
			if word=='<kb>':
				word_list.append('<kb>_'+str(kb_count))
				kb_count = kb_count+1
			else:
				word_list.append(word)	
		
		word_list = list(OrderedDict.fromkeys(word_list)) 
		if '</s>' in word_list:
                   word_list.remove('</s>')
            	if '</e>' in word_list:
                   word_list.remove('</e>')
            	if '<pad>' in word_list:
                   word_list.remove('<pad>')
            	if '<unk>' in word_list:
                   word_list.remove('<unk>')
		if '|' in ent:
			ent = ent.strip().split('|')
		else:
			ent = [x.strip() for x in ent.strip().split(',')]	
		pred = replace_kb_ent_in_resp(ent, word_list)
		print pred

		 
