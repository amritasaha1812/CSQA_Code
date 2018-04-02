import pattern.en
import re
import sys
import string
import traceback
import json
import nltk
from nltk.corpus import stopwords
import gensim
from annoy import AnnoyIndex
import os
import cPickle as pkl
stop = set(stopwords.words('english'))

types = {}
good_types = set([])
for x in json.load(open('/dccstor/cssblr/vardaan/dialog-qa/prop_obj_90_map5.json')).values():
	good_types.update(x)
for x in json.load(open('/dccstor/cssblr/vardaan/dialog-qa/prop_sub_90_map5.json')).values():
	good_types.update(x)
for k,v in json.load(open('/dccstor/cssblr/vardaan/dialog-qa/child_par_dict_name_2_corr.json')).items():
	if k not in good_types:
		continue
	v = set([x for x in v.lower().strip().split(' ') if x not in stop])
        plur_v = set([pattern.en.pluralize(vi) for vi in v])
        v = v.union(plur_v)
        for vi in v:
                if vi not in types:
                        types[vi] = []
		if k not in types[vi]:	
	                types[vi].append(k)
for k,v in json.load(open('/dccstor/cssblr/vardaan/dialog-qa/type_set_dict.json')).items():
	if k not in good_types:
		continue	
        v = set([x for x in v.lower().strip().split(' ') if x not in stop])
        plur_v = set([pattern.en.pluralize(vi) for vi in v])
	v = v.union(plur_v)
	for vi in v:
		if vi not in types:
			types[vi] = []
		if k not in types[vi]:	
			types[vi].append(k)
json.dump(types, open('annoy_index_type/type_names.json','w'), indent=1)
sys.exit(1)
word2vec_pretrain_embed = gensim.models.Word2Vec.load_word2vec_format('/dccstor/cssblr/amrita/resources/glove/GoogleNews-vectors-negative300.bin', binary=True)
f=300
index = AnnoyIndex(f, metric='euclidean')
index_desc = {}
count = 0
for word,ids in types.items():
	if word not in word2vec_pretrain_embed:
		print 'could not find ::::', word
		continue
	embed = word2vec_pretrain_embed[word]
	index.add_item(count, embed)
	index_desc[count] = ids
	count = count + 1
index.build(100)
index.save('annoy_index_type/glove_embedding_of_vocab.ann')
pkl.dump(index_desc, open('annoy_index_type/index2type.pkl','wb'))				


