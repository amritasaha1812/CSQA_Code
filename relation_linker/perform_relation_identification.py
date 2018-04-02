import re
import sys
import string
import traceback
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import os
import gensim
import cPickle as pkl
from annoy import AnnoyIndex
import json
wikidata_id_name_map={k:re.sub(r'[^\x00-\x7F]+',' ',v) for k,v in json.load(open('/dccstor/cssblr/vardaan/dialog-qa/item_data_filt.json')).items()}
glove_embedding = gensim.models.Word2Vec.load_word2vec_format('/dccstor/cssblr/amrita/resources/glove/GoogleNews-vectors-negative300.bin', binary=True)
print 'loaded glove embeddings'
ann = AnnoyIndex(300,  metric='euclidean')
ann.load('annoy_index_rel_noisy/glove_embedding_of_vocab.ann')
ann_pickle = pkl.load(open('annoy_index_rel_noisy/index2word.pkl'))
data_dir = '/dccstor/cssblr/vardaan/dialog-qa/QA_train_final5/'
count_files=0
rel_name_to_id = {}
for line in open('predicates_bw.tsv').readlines():
        line = line.strip().lower().split('\t')
        id = line[0]
        name = [x for x in ' '.join(line[1:]).split(' ') if x not in stop]
        for name_i in name:
		if name_i not in rel_name_to_id:	
			rel_name_to_id[name_i] = set([])
		rel_name_to_id[name_i].add(id)
for line in open('predicates_fw.tsv').readlines():
        line = line.strip().lower().split('\t')
        id = line[0]
        name = [x for x in ' '.join(line[1:]).split(' ') if x not in stop]
        for name_i in name:
                if name_i not in rel_name_to_id:
                        rel_name_to_id[name_i] = set([])
                rel_name_to_id[name_i].add(id)
ann_pickle_rel = {}
for id,name in ann_pickle.items():
	ann_pickle_rel[id] = rel_name_to_id[name]
pkl.dump(ann_pickle_rel, open('annoy_index_rel_noisy/index2rel.pkl','w'))
prec = 0.0
rec = 0.0
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
                    try:
                        utterance = utter['utterance'].lower()
                        utterance = re.sub(r'[^\x00-\x7F]+',' ',utterance)
                        utterance = str(utterance).translate(string.maketrans("",""),string.punctuation)
			if 'relations' in utter:
				relations = [x.lower() for x in utter['relations']]
			else:
				continue
                        if 'entities' in utter:
                                entities = utter['entities']
                        else:   
                                entities = []
                        if 'Qid' in utter:
                                entities.append(utter['Qid'])
                        if 'prop_Qid_par' in utter:
                                parent = utter['prop_Qid_par']
                        else:   
                                parent = None
                        entity_names = [wikidata_id_name_map[id].lower() for id in entities if id in wikidata_id_name_map]
                        if parent is not None and parent in wikidata_id_name_map:
                                parent_name = wikidata_id_name_map[parent].lower()
                        else:   
                                parent_name = None
                        utterance_replaced = utterance
                        for e in entity_names:
                                if e is not None:
                                        utterance_replaced = utterance_replaced.replace(e,'')
                        if parent_name is not None:
                                utterance_replaced = utterance_replaced.replace(parent_name, '')
                        utterance_replaced = re.sub(' +',' ', utterance_replaced)
                        words = set([x for x in utterance_replaced.split(' ') if not x.isdigit() and len(x)>1 and x in glove_embedding])
                        words = words - stop
			rel_ids = set([])
			for word in words:
				if word not in glove_embedding:
					continue			
				word_vec = glove_embedding[word]
				nns = ann.get_nns_by_vector(word_vec, 2)		
				for nn in nns:
					nn_word = ann_pickle[nn]
					rel_ids.update(rel_name_to_id[nn_word])
			#print 'predicted rel ids ', rel_ids
			#print 'true rel ids ', relations
			true_rel_ids = rel_ids.intersection(relations)
			if len(relations)>0:
				prec += float(len(true_rel_ids))/float(len(relations))
			if len(rel_ids)>0:
				rec += float(len(relations))/float(len(rel_ids))
			count_acc +=1.	
			if count_acc % 1000==0:
				print 'Prec ' , float(prec)/float(count_acc), ' over ', int(count_acc)
				print 'Rec ', float(rec)/float(count_acc), ' over ', int(count_acc)		
                    except:
                        #print 'error in utterance '
                        traceback.print_exc(file=sys.stdout)
                        continue	
print 'Prec ' , float(prec)/float(count_acc)
print 'Rec ', float(rec)/float(count_acc)
