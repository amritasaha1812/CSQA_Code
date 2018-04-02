import string
from string import maketrans
import re
import sys
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
print 'loaded wikidata_id_name_map'
relations = {}
for line in open('predicates_bw.tsv').readlines():
	line = line.strip().lower().split('\t')	
	rel = line[0]
	label = [x for x in ' '.join(line[1:]).split(' ') if x not in stop]
	for w in label:
		if w not in relations:
			relations[w] = set([])
		else:
			relations[w].add(rel)
for line in open('predicates_fw.tsv').readlines():
        line = line.strip().lower().split('\t')
        rel = line[0]
        label = [x for x in ' '.join(line[1:]).split(' ') if x not in stop]
        for w in label:
                if w not in relations:
                        relations[w] = set([])
                else:
                        relations[w].add(rel)	
all_relation_words = set([])	
all_relation_words.update(relations.keys())	
word2vec_pretrain_embed = gensim.models.Word2Vec.load_word2vec_format('/dccstor/cssblr/amrita/resources/glove/GoogleNews-vectors-negative300.bin', binary=True)
f=300
index = AnnoyIndex(f, metric='euclidean')
index_desc = {}
count = 0
for word in all_relation_words:
        word = word
        if word in word2vec_pretrain_embed:
                embed = word2vec_pretrain_embed[word]
                index.add_item(count, embed)
                index_desc[count] = word
                count = count+1
index.build(100)
index.save('annoy_index_noisy/glove_embedding_of_vocab.ann')
pkl.dump(index_desc, open('annoy_index_noisy/index2word.pkl','wb'))	

