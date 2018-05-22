from collections import OrderedDict
import re
import sys
from words2number import *
gold_file = "../"+sys.argv[2]+"/test_output_"+sys.argv[1]+"/true_sent.txt"#sys.argv[1]
pred_file = "../"+sys.argv[2]+"/test_output_"+sys.argv[1]+"/pred_sent.txt" #sys.argv[2]
goldlines = open(gold_file).readlines()
predlines = open(pred_file).readlines()
#entlines = open(ent_file).readlines()

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

def parse(line):
		line = line.replace('respectively','').strip()	
		number1 = None
                entity1 = None
                number2 = None
                entity2 = None
                try:
                        number1=text2int(line)
                        #full line was a single numerical (in words)
                except:
                        if line.isdigit():
                                number1 = int(line)
                                #full line was a single numerical (in digits)
                        else:
				line = ' '+line.strip()+' '
                                if ' and ' in line:
                                        parts = [x.strip() for x in line.split(' and ')]
					if len(parts)>2:
						print 'not handled ', line
					try:	
	                                        number1 = int(parts[0].split(' ')[0])
        	                                entity1 = " ".join(parts[0].split(' ')[1:])
					except:
						try:
							number1 = int(parts[0])
						except:
							try:
								number1 = text2int(parts[0])
							except:
								try:
									number1 = text2int(parts[0].split(' ')[0])
			                                                entity1 = " ".join(parts[0].split(' ')[1:])	
								except:
									if line.isdigit():
										number1 = int(line)
									else:
										number1 = None
										entity1 = None		
					try:
	                                        number2 = int(parts[1].split(' ')[0])
        	                                entity2 = " ".join(parts[1].split(' ')[1:])
					except:
						try:
                                                        number2 = int(parts[1])
                                                except:
							try:
								number2 = text2int(parts[1])
							except:
								try:
									number2 = text2int(parts[1].split(' ')[0])
									entity2 = " ".join(parts[1].split(' ')[1:]) 
								except:
	        	                                                if line.isdigit():
        	        	                                                number2 = int(line)
                                		                        else:
                                                	                	number2 = None
		                						entity2 = None	
                                else:
					line = line.strip()
					try:
						number1 = int(line.split(' ')[0])
        	                                entity1 = " ".join(line.split(' ')[1:])
					except:
						try:
							number1 = text2int(line.split(' ')[0])
							entity1 = " ".join(line.split(' ')[1:])
						except:
		                                        print 'not handled ', line
		return number1, entity1, number2, entity2

def parse2(line):
	return [x for x in line.strip().split(' ') if x.isdigit()]
prec = 0.0
rec = 0.0
jacc = 0.0
count = 0.0	
for goldline, predline in zip(goldlines,predlines):	
		goldline = goldline.strip().lower()
		predline = predline.strip().lower().replace('you mean','').replace('you and','').replace('you','').replace('?','').strip()
		predline = predline.replace('</s>','').replace('<kb>','')
		if len(goldline.strip())==0 or 'did you mean' in goldline:
			continue
		word_list = predline.strip().split(' ')
                word_list = list(OrderedDict.fromkeys(word_list)) 
                if '</s>' in word_list:
                   word_list.remove('</s>')
                if '</e>' in word_list:
                   word_list.remove('</e>')
                if '<pad>' in word_list:
                   word_list.remove('<pad>')
                if '<unk>' in word_list:
                   word_list.remove('<unk>')
		predline = ' '.join(word_list)
		gold_number1, gold_entity1, gold_number2, gold_entity2 = parse(goldline)
		pred_number1, pred_entity1, pred_number2, pred_entity2 = parse(predline)
		#acc_old = acc
		gold_set = parse2(goldline)#[x for x in [str(gold_number1), str(gold_number2)] if x!='None']
		pred_set = parse2(predline)#[x for x in [str(pred_number1), str(pred_number2)] if x!='None']
		ints = float(len([x for x in pred_set if x in gold_set]))
		union = ints + float(len([x for x in pred_set if x not in gold_set])) + float(len([x for x in gold_set if x not in pred_set]))
		#print 'pred_line ', predline, 'gold_set ', gold_set, 'pred_set ', pred_set,' ints ', ints, 'union ', union	
		if union>0:
			jacc += ints/union
		if len(pred_set)>0:
			prec += ints/float(len(pred_set))
		if len(gold_set)>0:
			rec += ints/float(len(gold_set))
		''''	
		if gold_number1==pred_number1 and gold_entity1==pred_entity1 and gold_number2==pred_number2 and gold_entity2==pred_entity2:
                        acc=acc+1.0
                elif gold_number1==pred_number2 and gold_entity1==pred_entity2 and gold_number2==pred_number1 and gold_entity2==pred_entity1:
                        acc=acc+1.0
                elif gold_number1 is not None and gold_entity1 is not None and gold_number1==pred_number1 and gold_entity1==pred_entity1 and gold_number2 is None and gold_entity2 is None:
                        acc=acc+1.0
                elif gold_number1 is not None and gold_entity1 is not None and gold_number1==pred_number1 and gold_entity1==pred_entity1 and gold_number2 is not None and gold_entity2 is not None:# and sys.argv[1]=='15':
                        acc=acc+0.5
                elif gold_number1 is not None and gold_entity1 is not None and gold_number1==pred_number2 and gold_entity1==pred_entity2 and gold_number2 is None and gold_entity2 is None:
                        acc=acc+1.0
                elif gold_number1 is not None and gold_entity1 is not None and gold_number1==pred_number2 and gold_entity1==pred_entity2 and gold_number2 is not None and gold_entity2 is not None:# and sys.argv[1]=='15':
                        acc=acc+0.5
                elif gold_number1==pred_number1 and gold_entity1==pred_entity1 and gold_number2 is None:
                        acc=acc+1.0
                elif gold_number1==pred_number2 and gold_entity1==pred_entity2 and gold_number2 is None:
                        acc=acc+1.0
		gold_parsed = str(gold_entity1)+'('+str(gold_number1)+') '+str(gold_entity2)+'('+str(gold_number2)+') '		
		pred_parsed = str(pred_entity1)+'('+str(pred_number1)+') '+str(pred_entity2)+'('+str(pred_number2)+') '
		#if acc-acc_old>0:
		print goldline, '::', gold_parsed,'  |||  ',predline, '::', pred_parsed, '  ||| ', (acc-acc_old)
		#print ''
		'''
		count=count+1.0
prec /= count
rec /= count
jacc /= count
f1 = (2*prec*rec)/(prec+rec)
print 'total prec ', prec, ' total rec ', rec, 'total jacc ', jacc, ' count ', count
print 'precsion ', prec
print 'recall ', rec
print 'jacc ', jacc
print 'f1 ', f1
