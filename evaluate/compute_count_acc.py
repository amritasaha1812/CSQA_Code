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
                #if entity1 is None:
                #        print line, ' ---->', number1
                #else:
                #        print line, ' ---->', entity1, '(',number1,') ',entity2, '(',number2,') '
		return number1, entity1, number2, entity2
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
		#print goldline+' || '+predline	
		gold_number1, gold_entity1, gold_number2, gold_entity2 = parse(goldline)
		pred_number1, pred_entity1, pred_number2, pred_entity2 = parse(predline)
		print gold_number1, '::',gold_entity1, ' ::',gold_number2, '::',gold_entity2
		print pred_number1, '::',pred_entity1, '::',pred_number2, '::',pred_entity2
		acc_old = acc
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
		#elif gold_number1 is not None  and (gold_number1==pred_number1 or gold_number1==pred_number2) and gold_number2 is None and gold_entity1 is None:# and sys.argv[1]=='15':
		#	acc=acc+0.5
		#elif gold_number1 is not None  and (gold_number1==pred_number1 or gold_number1==pred_number2) and gold_number2 is None and gold_entity1 is not None:# and sys.argv[1]=='15':
                #        acc=acc+0.5
		#elif gold_entity1 is not None  and (gold_entity1==pred_entity1 or gold_entity1==pred_entity2) and gold_number2 is None:# and sys.argv[1]=='15':
                #        acc=acc+0.5
		#elif gold_number1 is not None  and (gold_number1==pred_number1 or gold_number1==pred_number2) and gold_number2 is not None:# and sys.argv[1]=='15':
                #        acc=acc+0.25
                #elif gold_entity1 is not None  and (gold_entity1==pred_entity1 or gold_entity1==pred_entity2) and gold_number2 is not None:# and sys.argv[1]=='15':
                #        acc=acc+0.25
		#elif gold_number2 is not None  and (gold_number2==pred_number1 or gold_number2==pred_number2):# and sys.argv[1]=='15':
                #        acc=acc+0.25
                #elif gold_entity2 is not None  and (gold_entity2==pred_entity1 or gold_entity2==pred_entity2):# and sys.argv[1]=='15':
                #        acc=acc+0.25		
		#if (acc-acc_old)>0:
		'''
		if gold_entity1 is None:
			gold_entity1 = "None"		
		if gold_number1 is None:
			gold_number1 = "None"
		if gold_entity2 is None:
			gold_entity2 = "None"
		if gold_number2 is None:
			gold_number2 = "None"
		'''	
		gold_parsed = str(gold_entity1)+'('+str(gold_number1)+') '+str(gold_entity2)+'('+str(gold_number2)+') '		
		pred_parsed = str(pred_entity1)+'('+str(pred_number1)+') '+str(pred_entity2)+'('+str(pred_number2)+') '
		#if acc-acc_old>0:
		#print goldline, '::', gold_parsed,'  |||  ',predline, '::', pred_parsed, '  ||| ', (acc-acc_old)
		#print ''
		count=count+1.0
print acc/count
