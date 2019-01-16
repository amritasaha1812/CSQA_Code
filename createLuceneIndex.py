### TO RUN: python createLuceneIndex.py  <wikidata_dir> <transe_dir>

import cPickle as pkl
import re
import string
import json
import os
import lucene
from lucene import *
import codecs
from java.io import File
import sys
import nltk
import unidecode
import unicodedata
from nltk.corpus import stopwords
import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.util import Version
stop = set(stopwords.words('english'))
wikidata_dir = sys.argv[1]
transe_dir = sys.argv[2]

string.punctuation='!"#$&\'()*+,-./:;<=>?@[\]^_`{|}~ '
regex = re.compile('[%s]' % re.escape(string.punctuation))
lucene.initVM(vmargs=['-Djava.awt.headless=true'])
index_dir = os.path.join(transe_dir+'/lucene_index')
analyzer = StandardAnalyzer(Version.LUCENE_36)
index = SimpleFSDirectory(File(index_dir))
if not os.path.exists(index_dir):
        os.makedirs(index_dir)
config = IndexWriterConfig(Version.LUCENE_36, analyzer)
writer = IndexWriter(index, config)
with codecs.open(wikidata_dir+'/items_wikidata_n.json','r','utf-8') as data_file:
	item_data = json.load(data_file)
filtered_wikidata = pkl.load(open(transe_dir+'/ent_id_map.pickle'))
item_data = {k:v for k,v in item_data.items() if k in filtered_wikidata}
i=0
num_errors=0
for k,v in item_data.items():
	k = k.strip()
	doc = Document()
	v_orig2 = v
	v = unicodedata.normalize('NFKD', v).encode('ascii','ignore')
	v_orig = v.strip()
        v = v.lower().strip()
	doc.add(Field("wiki_id", str(k), Field.Store.YES, Field.Index.NOT_ANALYZED))
	doc.add(Field("wiki_name_orig", str(v_orig), Field.Store.YES, Field.Index.NOT_ANALYZED))
	doc.add(Field("wiki_name", str(v), Field.Store.YES, Field.Index.NOT_ANALYZED))
	doc.add(Field("wiki_name_analyzed", str(v), Field.Store.YES, Field.Index.ANALYZED))
	v_punct_removed = re.sub(' +', ' ', regex.sub(' ', v)).strip()
	doc.add(Field("wiki_name_analyzed_nopunct", str(v_punct_removed), Field.Store.YES, Field.Index.ANALYZED))
	v_stop_removed = " ".join([x for x in nltk.word_tokenize(v_punct_removed) if x not in stop])
	doc.add(Field("wiki_name_analyzed_nopunct_nostop", str(v_stop_removed), Field.Store.YES, Field.Index.ANALYZED))
	writer.addDocument(doc)
	i=i+1
	if i%10000==0:
		print 'finished ',i
print 'num errors while indexing ', num_errors
writer.close()
index.close()

