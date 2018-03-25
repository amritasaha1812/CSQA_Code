from collections import OrderedDict
import sys, re
sys.path.append('../')
from itertools import izip
import time
from load_wikidata_wfn import *
# from parse_active_set import *

states = [3,4,5,6,16,17,18,19]
test_types = ['hard']
#ks = [2,5,10,20]

#act_set_file = 'test_easy_active_set_-1.txt'
#pred_ent_name_file = 'top20_ent_id_from_mem.txt'
#k = 20

def parse_active_set(active_set, target):
    active_set = active_set.strip()
    anding = False
    orring = False
    notting1 = False
    notting2 = False    
    if active_set.startswith('AND(') or active_set.startswith('OR('):
        if active_set.startswith('AND('):
            anding = True
            active_set = re.sub('^\(|\)$','',active_set.replace('AND', '',1))
        if active_set.startswith('OR('):
            anding = True
            active_set = re.sub('^\(|\)$','',active_set.replace('OR', '',1))
        while active_set.startswith('(') and active_set.endswith(')'):
            active_set = re.sub('^\(|\)$','',active_set)
        active_set_parts = active_set.split(', ')
        active_set_part1 = active_set_parts[0].strip()
        active_set_part2 = active_set_parts[1].strip()  
        if active_set_part1.startswith('NOT('):
            active_set_part1 = re.sub('^\(|\)$','',active_set_part1.replace('NOT','',1))                
            notting1 = True
        is_present1 = parse_basic_active_set(active_set_part1.strip(), target)
        if notting1:
            is_present1 = not is_present1
        if active_set_part2.startswith('NOT('):
            active_set_part2 = re.sub('^\(|\)$','',active_set_part2.replace('NOT','',1))                    
            notting2 = True
        is_present2 = parse_basic_active_set(active_set_part2.strip(), target)
        if notting2:
            is_present2 = not is_present2
        if anding:
            if is_present1 and is_present2:
                return True
            else:
                return False
        if orring:
            if is_present1 or is_present2:
                return True
            else:
                return False
    else:
        return parse_basic_active_set(active_set, target)
                        
def parse_basic_active_set(active_set, target):
    # st_time = time.time()
    if len(active_set) == 0:
        return False
    active_set_orig = active_set

    while active_set.startswith('(') and active_set.endswith(')'):
        active_set = re.sub('^\(|\)$','',active_set)
    while active_set.startswith('(') and not active_set.endswith(')'):
        active_set = re.sub('^\(','',active_set)
    while active_set.endswith(')') and not active_set.startswith('('):
        active_set = re.sub('\)$','',active_set)

    assert not active_set.startswith('(')
    assert not active_set.endswith(')')

    # print 'time taken for regex proc = %f' % (time.time() - st_time)

    parent = None
    parent1 = None
    parent2 = None

    try:
        assert len(active_set.strip().split(',')) == 3
    except:
        print 'active_set = %s' % active_set_orig
        logging.exception('Something aweful happened')
        raise Exception('ERROR!!!')
    parts = active_set.strip().split(',')

    if parts[0].startswith('c') and parts[2].startswith('c'):
        parent1 = parts[0].split('(')[1].split(')')[0].strip()
        parent2 = parts[2].split('(')[1].split(')')[0].strip()
    if parts[0].startswith('c'):
        parent = parts[0].split('(')[1].split(')')[0].strip()
        ent = parts[2].strip()
    elif parts[2].startswith('c'):
        parent = parts[2].split('(')[1].split(')')[0].strip()
        ent = parts[0].strip()
    rel = parts[1].strip()
    if parent and ent:
        try:
            if child_par_dict[target] == parent and (target in wikidata[ent][rel] or ent in wikidata[target][rel] or target in reverse_dict[ent][rel] or ent in reverse_dict[target][rel]):
                # print 'time taken = %f' % (time.time() - st_time)
                return True
            else:
                # print 'time taken = %f' % (time.time() - st_time)
                return False
        except: 
            return False
    elif parent1 and parent2:
        try:
            if child_par_dict[target] == parent1:
                children2 = par_child_dict[parent2]
                for ent in children2:
                    if target in wikidata[ent][rel] or ent in wikidata[target][rel] or target in reverse_dict[ent][rel] or ent in reverse_dict[target][rel]:
                        # print 'time taken = %f' % (time.time() - st_time)
                        return True
            elif child_par_dict[target] == parent2:
                children1 = par_child_dict[parent1] 
                for ent in children1:
                    if target in wikidata[ent][rel] or ent in wikidata[target][rel] or target in reverse_dict[ent][rel] or ent in reverse_dict[target][rel]:
                        # print 'time taken = %f' % (time.time() - st_time)
                        return True       
        except:
            # print 'time taken = %f' % (time.time() - st_time)
            return False
        # print 'time taken = %f' % (time.time() - st_time)
        return False


def is_contained_in_act_set(active_set, ent_id):
    active_set_tokens = active_set.split('#')

    for active_set_token in active_set_tokens:
        try:
            start_time = time.time()
            if parse_active_set(active_set_token, ent_id):
                return True
            # print 'active_set = %s' % active_set_token
            # print 'time taken = %f' % (time.time() - start_time)
        except:
            print active_set_token
            logging.exception('Something aweful happened')
            raise Exception('ERROR!!!')
    return False
'''
for state in states:
    for test_type in test_types:	
	for k in ks:
		n_prec_sum = 0
		count = 0
		sTime= time.time()
		act_set_file = sys.argv[1]+'/test_output_'+test_type+'_'+str(state)+'/active_set.txt'
		pred_ent_name_file = sys.argv[1]+'/test_output_'+test_type+'_'+str(state)+'/top20_ent_id_from_mem.txt'
		with open(act_set_file) as act_set_lines, open(pred_ent_name_file) as pred_lines:
		    for act_set_line, pred_line in izip(act_set_lines,pred_lines):
		        if time.time() - sTime > 1000:
		            break
		        try:
		            if count % 1000 == 0:
                		print count
		            active_set = act_set_line.rstrip()
		            pred_entities = pred_line.rstrip().split(', ') # QIDs
        
		            pred_entities = pred_entities[:k]
		            n_topK = len([x for x in pred_entities if is_contained_in_act_set(active_set, x)])
		            n_prec_sum += n_topK*1.0/float(len(pred_entities))
		            count += 1
		        except:
		            break

		avg_prec = n_prec_sum*1.0/float(count)
		print 'File ', pred_ent_name_file
		print 'Avg. prec for k= ',k,'= %f' % avg_prec
		print ''
'''

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
	
for state in states:
        for test_type in test_types:
                n_prec_sum = 0
                count = 0
                sTime= time.time()
                act_set_file = sys.argv[1]+'/test_output_'+test_type+'_'+str(state)+'/active_set.txt'
                pred_ent_name_file = sys.argv[1]+'/test_output_'+test_type+'_'+str(state)+'/top20_ent_id_from_mem.txt'
                pred_file =sys.argv[2]+'/test_output_'+test_type+'_'+str(state)+'/pred_sent.txt'
                with open(act_set_file) as act_set_lines, open(pred_ent_name_file) as ent_lines, open(pred_file) as pred_lines:
                        for pred, ent, act_set_line in izip(pred_lines, ent_lines, act_set_lines):
                                word_list = pred.strip().split(' ')
                                kb_count = 1
                                for word in pred.strip().split(' '):
                                        if word=='<kb>':
                                                word_list.append('<kb>_'+str(kb_count))
                                                kb_count = kb_count+1
                                        else:   
                                                word_list.append(word)
                                word_list = list(OrderedDict.fromkeys(word_list))
                                if '|' in ent:
                                        ent = ent.strip().split('|')
                                else:   
                                        ent = [x.strip() for x in ent.strip().split(',')]
                                top_k_ent = replace_kb_ent_in_resp(ent, word_list)
                                if time.time() - sTime > 10000:
                                        continue
                                 #try:
                                if count % 1000 == 0:
                                        print count
                                active_sets = act_set_line.rstrip().split('#')
	                        pred_entities = top_k_ent
	                        if(len(pred_entities))>0:
					n_topK = 0
					for x in pred_entities:
						for active_set in active_sets:
							if is_contained_in_act_set(active_set, x):
								n_topK += 1
	                                n_prec_sum += n_topK*1.0/float(len(pred_entities))
                                count += 1
                                #except: 
                                #        continue
                        avg_prec = n_prec_sum*1.0/float(count)
                        print 'File ', pred_ent_name_file
                        print 'Avg. prec = %f' % avg_prec	
