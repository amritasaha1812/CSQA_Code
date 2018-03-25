from collections import OrderedDict
import re
import sys
from words2number import *
gold_file = "model_softmax_kvmem_validtrim_unfilt_newversion/test_output_"+sys.argv[2]+"_"+sys.argv[1]+"/gold_resp.txt"#sys.argv[1]
pred_file = "model_softmax_decoder_newversion/test_output_"+sys.argv[2]+"_"+sys.argv[1]+"/pred_sent.txt" #sys.argv[2]
ent_file = "model_softmax_kvmem_validtrim_unfilt_newversion/test_output_"+sys.argv[2]+"_"+sys.argv[1]+"/top20_ent_from_mem.txt"
goldlines = open(gold_file).readlines()
predlines = open(pred_file).readlines()
entlines = open(ent_file).readlines()

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

acc = 0.0
count = 0.0
for goldline, predline,entline in zip(goldlines,predlines,entlines):
                goldline = goldline.strip().lower()
                predline = predline.strip().lower()
                word_list = predline.strip().split(' ')
                '''           
                kb_count = 1
                for word in pred.strip().split(' '):
                        if word=='<kb>':
                                word_list.append('<kb>_'+str(kb_count))
                                kb_count = kb_count+1
                        else:
                                word_list.append(word)  
                '''
                word_list = list(OrderedDict.fromkeys(word_list))
                if '</s>' in word_list:
                   word_list.remove('</s>')
                if '</e>' in word_list:
                   word_list.remove('</e>')
                if '<pad>' in word_list:
                   word_list.remove('<pad>')
                if '<unk>' in word_list:
                   word_list.remove('<unk>')
                if '|' in entline:
                        entline = entline.strip().split('|')
                else:
                        entline = [x.strip() for x in entline.strip().split(',')]
                predline = replace_kb_ent_in_resp(entline, word_list)
		acc_old = acc
		print goldline,'||', predline,
		goldline = " ".join([x for x in goldline.lower().split(' ') if x in ['yes','no']])
		predline = " ".join([x for x in predline.lower().split(' ') if x in ['yes','no']])
		print '||',predline,
		if goldline==predline:
			acc=acc+1.0
		'''
		else:
			goldline = goldline.split(" ")
			predline = predline.split(" ")
			frac_acc = 0.0
			for i in range(min(len(goldline),len(predline))):
				if goldline[i]==predline[i]:
					frac_acc = frac_acc + 1.0/float(len(goldline))
			acc=acc+frac_acc
		'''
		print  '||',acc-acc_old
		count=count+1.0
		#print predline,'||',goldline
print acc/count
		
