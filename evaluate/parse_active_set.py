import sys
from load_wikidata2 import load_wikidata
import json
from itertools import izip
import re
wikidata, reverse_dict, prop_data, child_par_dict, wikidata_fanout_dict = load_wikidata() 
par_child_dict = json.load(open('/dccstor/cssblr/vardaan/dialog-qa/par_child_dict.json'))

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
	while active_set.startswith('(') and active_set.endswith(')'):
               active_set = re.sub('^\(|\)$','',active_set)
	while active_set.startswith('(') and not active_set.endswith(')'):
               active_set = re.sub('^\(','',active_set)
	print active_set
	parent = None
	parent1 = None
	parent2 = None	
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
	               if target in par_child_dict[parent] and (target in wikidata[ent][rel] or ent in wikidata[target][rel] or target in reverse_dict[ent][rel] or ent in reverse_dict[target][rel]):
				return True
	       	       else:
				return False
	       except:	
			return False
	elif parent1 and parent2:
		try:
			if target in par_child_dict[parent1]:
				children2 = par_child_dict[parent2]
				for ent in children2:
					if target in wikidata[ent][rel] or ent in wikidata[target][rel] or target in reverse_dict[ent][rel] or ent in reverse_dict[target][rel]:
						return True
			elif target in par_child_dict[parent2]:
				children1 = par_child_dict[parent1]	
                	  	for ent in children1:
                        	        if target in wikidata[ent][rel] or ent in wikidata[target][rel] or target in reverse_dict[ent][rel] or ent in reverse_dict[target][rel]:
                                	        return True	      
		except:
			return False
		
		return False


if __name__=="__main__":
	dir="/dccstor/cssblr/amrita/dialog_qa/code/hred_kvmem2_softmax/model_softmax_new/dump"
	target_file = dir+'/test_hard_target_'+sys.argv[1]+'.txt'
	active_set_file = dir+'/test_hard_active_set_'+sys.argv[1]+'.txt'	
	with open(target_file) as targetlines, open(active_set_file) as activelines:
		for target, active_set in izip(targetlines, activelines):
				target = target.strip().split('|')
				active_set = active_set.strip()
				for target_i in target:
					is_present = parse_active_set(active_set, target_i)
					print 'ACTIVE SET: ',active_set, ' TARGET: ',target_i, ' IS_PRESENT: ',is_present
