import codecs
import re
import nltk
from nltk.corpus import stopwords
import json
import string
import lucene
from lucene import *
from java.io import File
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader, IndexReader
from org.apache.lucene.index import Term
from org.apache.lucene.search import BooleanClause, BooleanQuery, PhraseQuery, TermQuery
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version
import unicodedata
import unidecode
stop = set(stopwords.words('english'))
string.punctuation='!"#$&\'()*+,-./:;<=>?@[\]^_`{|}~ '
regex = re.compile('[%s]' % re.escape(string.punctuation))

class LuceneSearch():
	def __init__(self,lucene_index_dir='lucene_index/', num_docs_to_return=100):
		lucene.initVM(vmargs=['-Djava.awt.headless=true'])
                directory = SimpleFSDirectory(File(lucene_index_dir))
                self.searcher = IndexSearcher(DirectoryReader.open(directory))
                self.num_docs_to_return =num_docs_to_return
                self.ireader = IndexReader.open(directory)
	
	def strict_search(self, value, value_orig=None):
		value_words = set(value.split(' '))
		if value_orig is None:
			value_orig = re.sub(' +', ' ', regex.sub(' ', value)).strip()
		else:
			value_orig = re.sub(' +', ' ', regex.sub(' ', value_orig)).strip()
		value = re.sub(' +', ' ', regex.sub(' ', value.lower())).strip()
                query = BooleanQuery()
		query.add(TermQuery(Term("wiki_name",value)), BooleanClause.Occur.SHOULD)
		query.add(TermQuery(Term("wiki_name",value_orig)), BooleanClause.Occur.SHOULD)
		query.add(TermQuery(Term("wiki_name_orig",value)), BooleanClause.Occur.SHOULD)
                query.add(TermQuery(Term("wiki_name_orig",value_orig)), BooleanClause.Occur.SHOULD)
		#print "0. query ",query
                scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs
		return scoreDocs
	
	def qid_search(self, value):
		query = TermQuery(Term("wiki_id",value))
		scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs
                return scoreDocs
	
	def search(self, words, words_orig, stopwords=[], min_length=0, slop=2, remove_digits=False, any_one_word_occur=False):
		words_without_digits = re.sub(r'\w*\d\w*', '', " ".join(words)).strip().split(" ")
		if remove_digits and len(words_without_digits)>0:
			words = words_without_digits
		words = [x for x in words if x.lower() not in stopwords and len(x)>min_length]
		words_orig = [x for x in words_orig if x.lower() not in stopwords and len(x)>min_length]
		
		if len(words)==0:
			return []
		query = BooleanQuery()
		query1 = PhraseQuery()
		query1.setSlop(slop)
		query2 = PhraseQuery()
                query2.setSlop(slop)
		query3 = PhraseQuery()
                query3.setSlop(slop)
		for word in words:
			query2.add(Term("wiki_name_analyzed_nopunct", word))
			query3.add(Term("wiki_name_analyzed_nopunct_nostop", word))
		for word in words_orig:
			query1.add(Term("wiki_name_analyzed", word))
		query.add(query1, BooleanClause.Occur.SHOULD)
		query.add(query2, BooleanClause.Occur.SHOULD)
		query.add(query3, BooleanClause.Occur.SHOULD)
		#print "1. query ", query
		scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs
		if len(scoreDocs)>0:
			#self.printDocs(scoreDocs)
			return scoreDocs
		query = BooleanQuery()
		for word in words:
			query_word = BooleanQuery()
			query_word.add(TermQuery(Term("wiki_name_analyzed_nopunct", word)), BooleanClause.Occur.SHOULD)
			query_word.add(TermQuery(Term("wiki_name_analyzed_nopunct_nostop", word)), BooleanClause.Occur.SHOULD)
			query.add(query_word, BooleanClause.Occur.MUST)
		#print "2. query ", query
		scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs
		if len(scoreDocs)>0:
			return scoreDocs
		query = BooleanQuery()
                for word in words_orig:
			query.add(TermQuery(Term("wiki_name_analyzed", word)), BooleanClause.Occur.MUST)
		#print "3. query ", query
		scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs	
		if len(stopwords)>0 and any_one_word_occur:
			query = BooleanQuery()
	                for word in words_orig:
        	                query.add(TermQuery(Term("wiki_name_analyzed", word)), BooleanClause.Occur.SHOULD)
		return scoreDocs
		
	def relaxed_search(self, value, text=None):
		value_orig = value.strip()
		value = re.sub(' +', ' ', regex.sub(' ', value.lower())).strip()
		if text is not None:
			text = re.sub(' +', ' ', regex.sub(' ', text.lower())).strip()
		words = nltk.word_tokenize(value)
		words_set = set(words)
                words_orig = nltk.word_tokenize(value_orig)
		if len(' '.join(words))==0:
			return []
		if len(words)==0:
                        return []
		scoreDocs = self.strict_search(value, value_orig)
		if len(scoreDocs)>0:
			wiki_entities = self.get_wiki_entities(scoreDocs, words_set, text)
			if len(wiki_entities)>0:
                        	return wiki_entities
		scoreDocs = self.search(words, words_orig, [])
		if len(scoreDocs)>0:
			wiki_entities = self.get_wiki_entities(scoreDocs, words_set, text)
                        if len(wiki_entities)>0:
                                return wiki_entities
		scoreDocs = self.search(words, words_orig, stop)
		if len(scoreDocs)>0:
                        wiki_entities = self.get_wiki_entities(scoreDocs, words_set, text)
                        if len(wiki_entities)>0:
                                return wiki_entities
		scoreDocs = self.search(words, words_orig, stop, 1)
		if len(scoreDocs)>0:
                        wiki_entities = self.get_wiki_entities(scoreDocs, words_set, text)
                        if len(wiki_entities)>0:
                                return wiki_entities
                return []
	
	def more_relaxed_search(self, value, text):
		wiki_entities = self.relaxed_search(value, text)
		if len(wiki_entities)==0:
			value_orig = value.strip()
	                value = re.sub(' +', ' ', regex.sub(' ', value.lower())).strip()
			if text is not None:
	        	        text = re.sub(' +', ' ', regex.sub(' ', text.lower())).strip()
			words = nltk.word_tokenize(value)
	                words_set = set(words)
        	        words_orig = nltk.word_tokenize(value_orig)
			scoreDocs = self.search(words, words_orig, stop, 1, 3)
			if len(scoreDocs)>0:
				return self.get_wiki_entities(scoreDocs, words_set, text)
			else:
				scoreDocs = self.search(words, words_orig, stop, 1, 3, True)
				if len(scoreDocs)>0:
                        	        return self.get_wiki_entities(scoreDocs, words_set, text)
				else:
					scoreDocs = self.search(words, words_orig, stop, 1, 3, True, True)
	                                if len(scoreDocs)>0:
        	                                return self.get_wiki_entities(scoreDocs, words_set, text)
					else:
						return []
		else:
			return wiki_entities	

	def get_wiki_entities(self, scoreDocs, value_words, text=None):
		if len(scoreDocs)>100:
                        return []
		entities = []
		for scoreDoc in scoreDocs:
			doc = self.searcher.doc(scoreDoc.doc)
			wiki_id = doc['wiki_id']
			doc = doc['wiki_name_analyzed_nopunct']
			#print doc
			doc_words = set(doc.strip().split(' ')) #re.sub(' +', ' ', regex.sub(' ', doc.lower())).strip().split(' '))
			if text is None or doc.strip() in text:
				if wiki_id not in entities:
					entities.append(wiki_id)
					#print 'searching for ', value_words, '::',doc+"("+wiki_id+"), "
			'''			
			extra_words = doc_words - value_words
			extra_words = extra_words - stop
			#print 'searching for ', value_words, ':: doc',doc_words ,' extra ', extra_words
			if len(extra_words)<2:
				entities.append(wiki_id)
				try:
					print 'searching for ', value_words, '::',doc+"("+wiki_id+"), "
				except:
					continue
			'''
		return entities	

	def printDocs(self, scoreDocs):
		for scoreDoc in scoreDocs:
			doc = self.searcher.doc(scoreDoc.doc)
			for f in doc.getFields():
				print f.name(),':', f.stringValue(),',  '
				
			print ''
		print '-------------------------------------\n'
