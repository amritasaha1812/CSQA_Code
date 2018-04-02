import pattern.en
import re
import sys
import string
import traceback
import json
import nltk
from nltk.corpus import stopwords
from  annoy import AnnoyIndex
stop = set(stopwords.words('english'))
import os
import cPickle as pkl
import gensim
wikidata_id_name_map={k:re.sub(r'[^\x00-\x7F]+',' ',v) for k,v in json.load(open('/dccstor/cssblr/vardaan/dialog-qa/item_data_filt.json')).items()}
glove_embedding = gensim.models.KeyedVectors.load_word2vec_format('/dccstor/cssblr/amrita/resources/glove/GoogleNews-vectors-negative300.bin', binary=True)
print 'loaded glove embeddings'
ann = AnnoyIndex(300,  metric='euclidean')
ann.load('annoy_index_type/glove_embedding_of_vocab.ann')
ann_pickle_type = pkl.load(open('annoy_index_type/index2type.pkl'))

types = json.load(open('annoy_index_type/type_names.json'))
def get_type(active_set):
	types_in_active_set = set([])
	for x in active_set.split(','):
		if 'c(' in x:
			t = x.replace('c(','').replace(')','').strip().replace('(','').split('|')	
			types_in_active_set.update(t)	
	return types_in_active_set

def get_filtered_utterance(utterance, utter):
        if 'entities' in utter:
                entities = utter['entities']
        else:
                entities = []
        if 'Qid' in utter:
                entities.append(utter['Qid'])
        entity_names = [wikidata_id_name_map[id].lower() for id in entities if id in wikidata_id_name_map]
        utterance_replaced = utterance
        for e in entity_names:
                if e is not None:
                        utterance_replaced = utterance_replaced.replace(e,'')
        utterance_replaced = re.sub(' +',' ', utterance_replaced)
        words = set([x for x in utterance_replaced.split(' ') if not x.isdigit() and len(x)>1 and x in glove_embedding])
        words = words - stop
	return words

data_dir = '/dccstor/cssblr/vardaan/dialog-qa/QA_train_final6/' 
prec = 0.0
rec = 0.0
count_files = 0
count_acc = 0.0
for dir in os.listdir(data_dir):
        if 'txt' in dir or 'pickle' in dir or 'xls' in dir:
                continue
        print dir
        for dir2 in os.listdir(data_dir+'/'+dir):
             if 'txt' in dir2 or 'pickle' in dir2 or 'xls' in dir2:
                   continue
             for file in os.listdir(data_dir+'/'+dir+'/'+dir2):
                if not file.endswith('json'):
                        continue
                count_files+=1
                if count_files%100==0:
                        print 'finished ',count_files
                data = json.load(open(data_dir+'/'+dir+'/'+dir2+'/'+file))
                for utter in data:
		    #print ('active_set' in utter), 'utterance ', utter['utterance']
		    	
                    try:
			if 'active_set' not in utter:
	                        utterance = utter['utterance'].lower()
        	                utterance = re.sub(r'[^\x00-\x7F]+',' ',utterance)
                	        utterance = str(utterance).translate(string.maketrans("",""),string.punctuation)
				utterance_filtered = get_filtered_utterance(utterance, utter)
			else:
				gold_types = set([])
				for x in utter['active_set']:
					gold_types.update(get_type(x))
				predicted_types = set([])
				predicted_type_names = set([])
				for type in types:
					if type in utterance:
						predicted_types.update(types[type])
						predicted_type_names.add(type)
				if len(predicted_type_names)>=0:
					for word in utterance_filtered:
						sing_w = pattern.en.singularize(word)
						plur_w = pattern.en.pluralize(word)
						nn_words = set([])
						if sing_w in glove_embedding:
							nn_words.update(ann.get_nns_by_vector(glove_embedding[sing_w], 5))
						if plur_w in glove_embedding:
							nn_words.update(ann.get_nns_by_vector(glove_embedding[plur_w], 5))
						for nn in nn_words:
							predicted_types.update(ann_pickle_type[nn])
						
				ints = predicted_types.intersection(gold_types)
				#if len(ints)==0 and len(gold_types)>0:
				#	print gold_types,  '::::', predicted_types, ':::', predicted_type_names
				if len(gold_types)>0:
					prec += float(len(ints))/float(len(gold_types))	
				if len(predicted_types)>0:
					rec += float(len(ints))/float(len(predicted_types))
				if len(gold_types)>0:
					count_acc += 1.0
					if count_acc %1000==0:
						print 'Prec ', prec/count_acc, ' Recall ', rec/count_acc, ' Over ', int(count_acc)
				#utterance = None
				#gold_types = None
				#predicted_types = None
		    except:
				print traceback.print_exc()
				continue	
		    
print 'Prec ', prec/count_acc, ' Recall ', rec/count_acc, ' Over ', int(count_acc)	
