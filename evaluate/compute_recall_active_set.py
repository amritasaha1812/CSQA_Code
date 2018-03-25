from collections import OrderedDict
import sys, re
from itertools import izip
import time
#from load_wikidata_wfn import *
from parse_active_set import *
#par_child_dict = json.load(open('/dccstor/cssblr/vardaan/dialog-qa/par_child_dict.json'))
total_entities = len(par_child_dict)
states = [6,9,10,12,13,14,15]#,7,8]
test_types = ['hard']
ks = [2,5,10,20]

#act_set_file = 'test_easy_active_set_-1.txt'
#pred_ent_name_file = 'top20_ent_id_from_mem.txt'
#k = 20
def get_active_set_size(active_set):
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
        set_part1 = get_basic_active_set(active_set_part1.strip())
        if active_set_part2.startswith('NOT('):
            active_set_part2 = re.sub('^\(|\)$','',active_set_part2.replace('NOT','',1))
            notting2 = True
        set_part2 = get_basic_active_set(active_set_part2.strip())
        set_final = set([])
        set_final_len = 0
        if anding:
            if notting1 and not notting2:
                set_final = set_part2 - set_part1
                set_final_len = len(set_final)
            elif notting2 and not notting1:
                set_final = set_part1 - set_part2
                set_final_len = len(set_final)
            elif not notting1 and not notting2:
                set_final = set_part1.intersection(set_part2)
                set_final_len = len(set_final)
            elif notting1 and notting2:
                #print 'found notting1 and notting2 ', active_set       
                set_final.update(set_part1)
                set_final.update(set_part2)
                set_final_len = total_entities - len(set_final)
	if orring:
            if notting2 and not notting1:
                #set_final.update(set_part1)
                set_final = set_part1.intersection(set_part2)
                set_final_len = total_entities - len(set_part2) + len(set_final)
            elif notting1 and not notting2:
                set_final = set_part1.intersection(set_part2)
                set_final_len = total_entities - len(set_part1) + len(set_final)
            elif not notting1 and not notting2:
                set_final.update(set_part1)
                set_final.update(set_part2)
                set_final_len = len(set_final)
            elif notting1 and notting2:
                set_final= set_part1.intersection(set_part2)
                set_final_len = total_entities - len(set_final)
	return set_final_len
    else:
        return len(get_basic_active_set(active_set.strip()))

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
        if not notting1:
            is_present1 = not is_present1
        if active_set_part2.startswith('NOT('):
            active_set_part2 = re.sub('^\(|\)$','',active_set_part2.replace('NOT','',1))                    
            notting2 = True
        is_present2 = parse_basic_active_set(active_set_part2.strip(), target)
	set_final = set([])
	set_final_len = 0
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
                        
def get_basic_active_set(active_set):
    if len(active_set) == 0:
        return set([])
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
    feasible_children = set([])
    if parent and ent:
	if parent not in par_child_dict:
		return set([])			
	sources = set([])
	targets = set([])
	
        try:
		sources.update(wikidata[ent][rel])
		targets.update(wikidata[ent][rel])
		sources.update(reverse_dict[ent][rel])
		targets.update(reverse_dict[ent][rel])
	except:
		pass
	all_entities = set([])
	all_entities.update(sources)
	all_entities.update(targets)	
	children_of_par = par_child_dict[parent]
	feasible_children = set(children_of_par).intersection(all_entities)
	num_children = len(feasible_children)

    elif parent1 and parent2:
	if parent1 not in par_child_dict or parent2 not in par_child_dict:
		return set([])		
	children2 = par_child_dict[parent2]
	children1 = par_child_dict[parent1]
	if len(children1)<len(children2):
		for child in children1:
			if child in wikidata and rel in wikidata[child]:
				feasible_set.update(wikidata[child][re].intersection(children2))
	else:
		for child in children2:
			if child in wikidata and rel in wikidata[child]:
				feasible_set.update(wikidata[child][re].intersection(children1))
	num_children = len(feasible_children)				
	#print 'size of active set ', num_children
    return feasible_children
	
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
fw = open('out7.txt','w')
dir1='model_softmax_kvmem_validtrim_unfilt_newversion'
dir2='model_softmax_decoder_newversion'

for state in states:
        for test_type in test_types:
                n_prec_sum = 0
                count = 0
                sTime= time.time()
                act_set_file = dir1+'/test_output_'+test_type+'_'+str(state)+'/active_set.txt'
                pred_ent_name_file = dir1+'/test_output_'+test_type+'_'+str(state)+'/top20_ent_id_from_mem.txt'
                pred_file = dir2+'/test_output_'+test_type+'_'+str(state)+'/pred_sent.txt'
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
                                if time.time() - sTime > 100000:
					#print " time break"
                                        continue
                                #try:
                                #if count % 1000 == 0:
                                #        print count
			
				active_sets = act_set_line.rstrip().split('#')
				active_set_sizes = []
				for active_set in active_sets:	
                                	active_set_size = get_active_set_size(active_set)
					active_set_sizes.append(active_set_size)
				
	                        pred_entities = top_k_ent
				n_topK = 0
				for x in pred_entities:
					for i,active_set in enumerate(active_sets):
						if is_contained_in_act_set(active_set,x):
							n_topK +=1
							break
				active_set_size = sum(active_set_sizes)
                                n_topK = active_set_size - n_topK
				if active_set_size>0:
	                        	n_prec_sum += 1-(n_topK*1.0/float(active_set_size))
                                count += 1
                                #except: 
                                #        break
                        avg_prec = n_prec_sum*1.0/float(count)
                        fw.write('File '+pred_ent_name_file+'\n')
                        fw.write('Avg. prec = '+str(avg_prec)+'\n')
			print 'File '+pred_ent_name_file
			print 'Avg. prec = '+str(avg_prec)
