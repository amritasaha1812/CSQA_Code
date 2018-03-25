import json
import sys, os, lucene
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

class LuceneSearch():
	def __init__(self,lucene_index_dir='/dccstor/cssblr/amrita/dialog_qa/code/prepro_lucene/lucene_index/'):
		lucene.initVM(vmargs=['-Djava.awt.headless=true'])
                directory = SimpleFSDirectory(File(lucene_index_dir))
                self.searcher = IndexSearcher(DirectoryReader.open(directory))
                self.num_docs_to_return =5
                self.ireader = IndexReader.open(directory)
		
	def search(self, value):
		query = TermQuery(Term("wiki_name",value.lower()))
		#query = BooleanQuery()
		#query.add(new TermQuery(Term("wikidata_name",v)),BooleanClause.Occur.SHOULD)
		#query.add(new TermQuery(Term("wikidata_name",v)),BooleanClause.Occur.SHOULD)  
		scoreDocs = self.searcher.search(query, self.num_docs_to_return).scoreDocs
		for scoreDoc in scoreDocs:
			doc = self.searcher.doc(scoreDoc.doc)
			for f in doc.getFields():
				print f.name(),':', f.stringValue(),',  '
			print ''
		print '-------------------------------------\n'

if __name__=="__main__":
	ls = LuceneSearch()
	ls.search("United States")
	ls.search("India")
	ls.search("Barrack Obama")
	ls.search("Obama")
