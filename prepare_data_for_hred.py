import gensim
import traceback
import os
import pattern.en
import json
import copy
import cPickle as pkl
from annoy import AnnoyIndex
import argparse
import logging
import nltk
from nltk import word_tokenize
from itertools import izip
import collections
from collections import Counter
from LuceneSearch import *
from question_parser_lucene2 import QuestionParser
from clean_utils import read_file_as_dict
from text_util import clean_word
from data_utils import *
import random
import csv
import fnmatch
import codecs
from load_wikidata2 import load_wikidata
import unicodedata
import unidecode
import logging
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
from words2number import *
import json
MAX_RELEVANT_ENTITIES = 4
HOPS_FROM_QN_ENTITY = 1
MAX_CANDIDATE_ENTITIES = 10000
MAX_CANDIDATE_TUPLES = 100000
class PrepareData():
    def __init__(self, max_utter, max_len, start_symbol_index, end_symbol_index, unk_symbol_index, pad_symbol_index, pad_kb_symbol_index, nkb_symbol_index, stopwords, stopwords_histogram, lucene_dir, transe_dir, wikidata_dir, glove_dir, max_mem_size, max_target_size, vocab_max_len, all_possible_ngrams, cutoff=-1):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('prepare_data_for_hred')
        self.max_utter = max_utter
        self.max_len = max_len
        self.unknown_word_id = unk_symbol_index
        self.start_word_id = start_symbol_index
        self.pad_word_id = pad_symbol_index
        self.end_word_id = end_symbol_index
        self.kb_word_id = 4
        self.start_word_symbol = '</s>'
        self.end_word_symbol = '</e>'
        self.pad_symbol = '<pad>'
        self.unk_symbol = '<unk>'
        self.kb_word_symbol = '<kb>'
        self.pad_kb_symbol_index = pad_kb_symbol_index
	self.nkb_symbol_index = nkb_symbol_index
	self.pad_kb_symbol = '<pad_kb>'
	self.nkb_symbol = '<nkb>'
        self.cutoff = cutoff
	self.vocab_max_len = vocab_max_len
        self.all_possible_ngrams = all_possible_ngrams
        self.input = None
        self.output = None
        self.vocab_file = None
        self.vocab_dict = None
	self.response_vocab_file = None
	self.response_vocab_dict = None
        self.word_counter = None
        self.max_mem_size = max_mem_size
        self.max_target_size = max_target_size
	self.lemmatizer = nltk.WordNetLemmatizer()
        self.stemmer = nltk.stem.porter.PorterStemmer()
        self.bad_qids = set(['Q184386','Q1541554','Q540955','Q2620241','Q742391'])  #adding Yes/No
        self.bad_qids.update(pkl.load(open('wikidata_entities_with_digitnames.pkl')))
        self.wikidata_qid_to_name = json.load(open(wikidata_dir+'/items_wikidata_.json'))
	#Taken from Su Nam Kim Paper...
        self.grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
              
            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        self.chunker = nltk.RegexpParser(self.grammar)
        #************************************************************************#
	self.stop_set = pkl.load(open(stopwords))
        self.stop_vocab = read_file_as_dict(stopwords_histogram)
        self.ls = LuceneSearch(lucene_dir)
        self.question_parser = QuestionParser(None, self.stop_vocab, self.stop_set, self.bad_qids, self.ls, self.wikidata_qid_to_name, self.all_possible_ngrams)
        self.wikidata, self.reverse_dict, self.prop_data, self.child_par_dict, self.child_all_par_dict, self.wikidata_fanout_dict = load_wikidata(wikidata_dir)
        self.id_entity_map = {self.pad_kb_symbol_index:self.pad_kb_symbol, self.nkb_symbol_index: self.nkb_symbol}
        self.id_entity_map.update({(k+2):v for k,v in pkl.load(open(transe_dir+'/id_entity_map.pickle','rb')).iteritems()})
	
        self.id_rel_map = {self.pad_kb_symbol_index:self.pad_kb_symbol, self.nkb_symbol_index: self.nkb_symbol}
        self.id_rel_map.update({(k+2):v for k,v in pkl.load(open(transe_dir+'/id_rel_map.pickle','rb')).iteritems()})

        self.entity_id_map = {v: k for k, v in self.id_entity_map.iteritems()}
        self.rel_id_map = {v: k for k, v in self.id_rel_map.iteritems()}

        self.kb_ov_idx = 1 # symbol assigned to entries out of kb or for padding to target_ids
        self.kb_rel_ov_idx = 1 # symbol assigned to entries out of kb or for padding to target_ids

	glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_dir, binary=True)#/dccstor/cssblr/amrita/resources/glove/GoogleNews-vectors-negative300.bin', binary=True)
        vocab = glove_model.wv.vocab.keys()
        self.glove_embedding = {v:glove_model.wv[v] for v in vocab}
        print 'loaded glove embeddings'
	self.ann_rel = AnnoyIndex(300,  metric='euclidean')
	self.ann_rel.load('relation_linker/annoy_index_rel_noisy/glove_embedding_of_vocab.ann')
	self.ann_pickle_rel = pkl.load(open('relation_linker/annoy_index_rel_noisy/index2rel.pkl'))
	self.ann_type = AnnoyIndex(300,  metric='euclidean')
	self.ann_type.load('type_linker/annoy_index_type/glove_embedding_of_vocab.ann')
	self.ann_pickle_type = pkl.load(open('type_linker/annoy_index_type/index2type.pkl'))
	self.types = json.load(open('type_linker/annoy_index_type/type_names.json'))
	
        #************************************************************************#
    
    def safe_pickle(self, obj, filename):
        if os.path.isfile(filename):
            self.logger.info("Overwriting %s." % filename)
        else:
            self.logger.info("Saving to %s." % filename)

        with open(filename, 'wb') as f:
                pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)



    def prepare_data(self, input, vocab_file, vocab_stats_file, response_vocab_file, output, dialogue_pkl_file, ques_type_id = -1):
	if not os.path.isdir(input) or len(os.listdir(input))==0:
            raise Exception("Input file not file")
        self.vocab_file = vocab_file
        self.vocab_stats_file = vocab_stats_file
	self.response_vocab_file = response_vocab_file
        self.output = output
        self.dialogue_context_file = self.output+"_context_%d.txt" % int(ques_type_id)
        self.dialogue_target_file = self.output+"_target_%d.txt" % ques_type_id
	self.dialogue_response_file = self.output+"_response_%d.txt" % ques_type_id
	self.dialogue_orig_response_file = self.output+"_orig_response_%d.txt" % ques_type_id
        self.sources_file = self.output+"_sources_%d.txt" % ques_type_id
        self.relation_file = self.output+"_relation_%d.txt" % ques_type_id
        self.target_file = self.output+"_value_target_%d.txt" % ques_type_id
        self.active_set_file = self.output+"_active_set_%d.txt" % ques_type_id
        if os.path.isfile(vocab_file):
            print 'found pre-existing vocab file.. reusing it'
            create_vocab = False
        else:
            create_vocab = True
        self.read_jsondir(input, create_vocab, ques_type_id)

        #print 'json dir read complete'

        if create_vocab:
            self.build_vocab() 
        else:
            self.read_vocab()
        if create_vocab or not os.path.exists(dialogue_pkl_file):
            self.binarize_corpus(self.dialogue_context_file, self.dialogue_target_file, self.dialogue_response_file, self.dialogue_orig_response_file, self.active_set_file, self.sources_file, self.relation_file, self.target_file, dialogue_pkl_file)
        
    def read_jsondir(self, json_dir, create_vocab=False, ques_type_id = -1):
        if os.path.isfile(self.dialogue_context_file):
            os.remove(self.dialogue_context_file)
        if os.path.isfile(self.dialogue_target_file):
            os.remove(self.dialogue_target_file)
        if os.path.isfile(self.dialogue_response_file):
            os.remove(self.dialogue_response_file)
        if os.path.isfile(self.dialogue_orig_response_file):
            os.remove(self.dialogue_orig_response_file)
        if os.path.isfile(self.sources_file):
            os.remove(self.sources_file)
        if os.path.isfile(self.relation_file):
            os.remove(self.relation_file)
        if os.path.isfile(self.target_file):
            os.remove(self.target_file)
        if os.path.isfile(self.active_set_file):
            os.remove(self.active_set_file)

        if create_vocab:
            self.word_counter = Counter()
	    self.response_word_counter = Counter()
        else:
            self.word_counter = None
	    self.response_word_counter = None
        for root, dirnames, filenames in os.walk(json_dir):
            for filename in fnmatch.filter(filenames, '*.json'):
                self.read_jsonfile(os.path.join(root, filename), create_vocab, ques_type_id)
       

    def isint(self, x):
        try:
                int(x)
                return True
        except:
                try:
                        text2int(x.strip())
                        return True
                except:
                        return False
 
    def find_relation(self, utterance, entities):
        feasible_entities = set([e for e in entities if e in self.wikidata])
        relations = set([])
        words = [x for x in nltk.word_tokenize(utterance.strip().lower()) if not self.isQid(x) and not x.isdigit() and len(x)>1 and x not in stopwords and x in self.glove_embedding]
        for word in words:
                vec = self.glove_embedding[word]
                nns = self.ann_rel.get_nns_by_vector(vec, 5)
                for nn in nns:
                        nn_relations = self.ann_pickle_rel[nn]
                        for rel in nn_relations:
                                if len(entities)==0 or (any([rel in self.wikidata[e] for e in feasible_entities]) and rel in self.prop_data and rel in self.rel_id_map):
                                        relations.add(rel)
        return relations

    def find_type(self, utterance):
        types = set([])
        for type in self.types:
                if type in utterance:
                        types.update(self.types[type])
        words = [x for x in nltk.word_tokenize(utterance.strip().lower()) if not self.isQid(x) and not x.isdigit() and len(x)>1 and x not in stopwords and x in self.glove_embedding]
        for word in words:
                sing_w = pattern.en.singularize(word)
                plur_w = pattern.en.pluralize(word)
                nn_words = set([])
                if sing_w in self.glove_embedding:
                        nn_words.update(self.ann_type.get_nns_by_vector(self.glove_embedding[sing_w], 5))
                if plur_w in self.glove_embedding:
                        nn_words.update(self.ann_type.get_nns_by_vector(self.glove_embedding[plur_w], 5))
                for nn in nn_words:
                        types.update(self.ann_pickle_type[nn])
        return types


    def isQid(self,input_str):
        if input_str.upper() not in self.entity_id_map:
            return False

        if len(input_str) == 0:
            return False

        if input_str in ['<pad_kb>','<nkb>']:
            return True

        char0 = input_str[0]
        rem_chars = input_str[1:]

        if char0 != 'Q' and char0 !='q':
            return False

        try:
            x = int(rem_chars)
        except:
            return False
        return True 

    def read_jsonfile(self, json_file, create_vocab, ques_type_id = -1):
        #print 'json filename: %s' % json_file
        try:
            dialogue = json.load(codecs.open(json_file,'r','utf-8'))
        except:
            return None
        if dialogue is None or len(dialogue)==0:
            return None
        filter(None, dialogue)

        

        dialogue_vocab = {}
        dialogue_contexts = []
	dialogue_context_utterances  = []
        dialogue_targets = []
        dialogue_instance = []
	dialogue_responses = []
	dialogue_orig_responses = []

        dialog_sources = []
        dialog_relations = []
        dialog_targets = []
        dialog_active_sets = []

        utterance_count = 0

        for utter_id, utterance in enumerate(dialogue):
            # print utterance
            utterance_count += 1
            # print utterance_count
            
            if utterance is None:
                continue
            if not isinstance(utterance, dict):
                raise Exception('error in reading dialogue json')
                continue
            if len(utterance)==0:
                continue
            speaker = utterance['speaker']
	    nlg = utterance['utterance']
	    if isinstance(nlg, unicode):
		nlg = unicodedata.normalize('NFKD', nlg).encode('ascii','ignore')
	    else:
		nlg = unicodedata.normalize('NFKD', unicode(nlg, "utf-8")).encode('ascii','ignore')
            # print nlg
            if nlg is not None:
                   nlg = nlg.strip().encode('utf-8')
            if nlg is None:
                nlg = ""
            # nlg = nlg.lower().replace("|","")
            nlg = nlg.replace("|","")
            try:
                nlg_words = nltk.word_tokenize(nlg)
            except:
                nlg_words = nlg.split(" ")
            # if create_vocab:
            #     self.word_counter.update(nlg_words[:self.max_len-2])
            dialogue_instance.append(nlg)
            #_, context_list = self.question_parser.get_utterance_entities(nlg)
            # print 'flag'
            
            if create_vocab:
		_, context_list = self.question_parser.get_utterance_entities(nlg)
                self.word_counter.update([x for x in nltk.word_tokenize(context_list) if not self.isQid(x)]) # update vocab for non-Qid words
            is_ques_relevant = False
	    
            if ques_type_id == 16:
                if speaker=="SYSTEM" and 'ques_type_id' in dialogue[utter_id-1] and ('count_ques_sub_type' in dialogue[utter_id-1] and ((dialogue[utter_id-1]['ques_type_id'] == 7 and dialogue[utter_id-1]['count_ques_sub_type'] in [2,3]) or (dialogue[utter_id-1]['ques_type_id'] == 8 and dialogue[utter_id-1]['count_ques_sub_type'] in [3,4])) and ('active_set' in utterance)):
                    is_ques_relevant = True
            elif ques_type_id == 17:
                if speaker=="SYSTEM" and 'ques_type_id' in dialogue[utter_id-1] and ('count_ques_sub_type' in dialogue[utter_id-1] and ((dialogue[utter_id-1]['ques_type_id'] == 7 and dialogue[utter_id-1]['count_ques_sub_type'] in [1,5,7]) or (dialogue[utter_id-1]['ques_type_id'] == 8 and dialogue[utter_id-1]['count_ques_sub_type'] in [1,2,6,8])) and ('active_set' in utterance)):
                    is_ques_relevant = True
            elif ques_type_id == 18:
                if speaker=="SYSTEM" and 'ques_type_id' in dialogue[utter_id-1] and ('count_ques_sub_type' in dialogue[utter_id-1] and ((dialogue[utter_id-1]['ques_type_id'] == 7 and dialogue[utter_id-1]['count_ques_sub_type'] in [4,8]) or (dialogue[utter_id-1]['ques_type_id'] == 8 and dialogue[utter_id-1]['count_ques_sub_type'] in [5,9])) and ('active_set' in utterance)):
                    is_ques_relevant = True
            elif ques_type_id == 19:
                if speaker=="SYSTEM" and 'ques_type_id' in dialogue[utter_id-1] and ('count_ques_sub_type' in dialogue[utter_id-1] and ((dialogue[utter_id-1]['ques_type_id'] == 7 and dialogue[utter_id-1]['count_ques_sub_type'] in [6,9]) or (dialogue[utter_id-1]['ques_type_id'] == 8 and dialogue[utter_id-1]['count_ques_sub_type'] in [7,10])) and ('active_set' in utterance)):
                    is_ques_relevant = True
	    elif ques_type_id == 4:
		if speaker=="SYSTEM" and (('ques_type_id' in dialogue[utter_id-1] and dialogue[utter_id-1]['ques_type_id'] == ques_type_id) or ('ques_type_id' not in dialogue[utter_id-1] and utter_id > 2 and 'ques_type_id' in dialogue[utter_id-3] and dialogue[utter_id-3]['ques_type_id'] == ques_type_id)):			is_ques_relevant = True
	    elif ques_type_id == 3:
		if speaker=="SYSTEM" and (('ques_type_id' in dialogue[utter_id] and dialogue[utter_id]['ques_type_id'] == ques_type_id) or (utter_id > 1 and 'ques_type_id' in dialogue[utter_id-2] and dialogue[utter_id-2]['ques_type_id']== ques_type_id)):
		    is_ques_relevant = True
	    elif ques_type_id == 9:
		if speaker=="SYSTEM" and ('ques_type_id' in dialogue[utter_id-1] and dialogue[utter_id-1]['ques_type_id'] == 2 and 'sec_ques_sub_type' in dialogue[utter_id-1] and dialogue[utter_id-1]['sec_ques_sub_type'] in [2,3]) and ('active_set' in utterance):
		    is_ques_relevant = True
	    elif ques_type_id == 10:
		if speaker=="SYSTEM" and 'ques_type_id' in dialogue[utter_id-1] and (dialogue[utter_id-1]['ques_type_id'] == 1 or (dialogue[utter_id-1]['ques_type_id'] == 2 and 'sec_ques_sub_type' in dialogue[utter_id-1] and dialogue[utter_id-1]['sec_ques_sub_type'] in [1,4])) and ('active_set' in utterance):
		    is_ques_relevant = True						
            elif ques_type_id in [ -1, 6, 5]:
                if speaker=="SYSTEM" and (ques_type_id < 0 or ('ques_type_id' in dialogue[utter_id-1] and dialogue[utter_id-1]['ques_type_id'] == ques_type_id)) and ('active_set' in utterance or ques_type_id < 0):
                    is_ques_relevant = True
                # print 'flag 0'
                # last_utterance = dialogue_instance[-1]
            if is_ques_relevant:
		if ques_type_id < 0 and 'active_set' not in utterance:
			utterance['active_set'] = ''
                if 'active_set' not in utterance:
		    active_set = ''
		elif len(utterance['active_set']) == 0:
		    active_set = ''
                elif type(utterance['active_set'][0]) == type(list()):
                    active_set = '#'.join(['(%s,%s,%s)'% (x[0],x[1],x[2]) for x in utterance['active_set']])
                else:
                    active_set = '#'.join(utterance['active_set'])
                dialog_active_sets.append(active_set)

                padded_clipped_dialogue = self.pad_or_clip_dialogue(dialogue_instance)
                if len(padded_clipped_dialogue)!=(self.max_utter+1):
                    raise Exception('some problem with dialogue instance, len != max_utter+1')
                dialogue_instance_context = padded_clipped_dialogue[:-1]
                dialogue_instance_target = dialogue_instance[-1]
		dialogue_orig_responses.append(dialogue_instance_target)
               	#dialogue_responses.append(dialogue_instance_target) 
                # print 'flag 1'
                #target_entities = self.question_parser.get_NER(dialogue_instance_target) # target entities is a list of QIDs
                # print 'flag 2'
		if 'ans_list_full' in utterance:
			target_entities = utterance['ans_list_full']
		elif 'entities' in utterance:
			target_entities = utterance['entities']
		else:
			target_entities = []
                if len(target_entities) > 0:
                    dialogue_targets.append(get_str_of_seq(target_entities))
                else:
                    dialogue_targets.append('')

                def get_tuples_involving_entities(candidate_entities,relations_in_context=None, types_in_context=None):
		    tuples = set([])
                    pids = set([])
                    for QID in [q1 for q1 in candidate_entities if q1 in self.child_par_dict and q1 in self.entity_id_map]:
                        QID_type_matched = False
                        if types_in_context is None or (QID in self.child_all_par_dict and len(set(self.child_all_par_dict[QID]).intersection(types_in_context))>0):
                                QID_type_matched = True
                        feasible_pids = [p for p in self.wikidata[QID] if p in self.prop_data and p in self.rel_id_map]
                        if relations_in_context is not None:
                                detected_pids = set(feasible_pids).intersection(relations_in_context)
                                if len(detected_pids)==0:
                                        detected_pids = set(feasible_pids)
                        else:
                                detected_pids = set(feasible_pids)
                        pids.update(detected_pids)
                        for pid in detected_pids:
                            feasible_qids = set([q for q in self.wikidata[QID][pid] if q in self.entity_id_map and q in self.child_par_dict])
                            if types_in_context is None or QID_type_matched:
                                detected_qids = feasible_qids
                            else:
                                detected_qids = set([x for x in feasible_qids if len(set(self.child_all_par_dict[x]).intersection(types_in_context))>0])
                            if len(detected_qids)==0:
                                detected_qids = feasible_qids
                            for qid in detected_qids:
                                tuples.add((QID, pid, qid))
                                tuples.add((qid, pid, QID))    
                    for QID in [q1 for q1 in candidate_entities if q1 in self.reverse_dict and q1 in self.entity_id_map]:
                        QID_type_matched = False
                        if types_in_context is None or (QID in self.child_all_par_dict and len(set(self.child_all_par_dict[QID]).intersection(types_in_context))>0):
                                QID_type_matched = True
                        feasible_pids = [p for p in self.reverse_dict[QID] if p in self.prop_data and p in self.rel_id_map]
                        if relations_in_context is not None:
                                detected_pids = set(feasible_pids).intersection(relations_in_context)
                                if len(detected_pids)==0:
                                        detected_pids = set(feasible_pids)
                        else:
                                detected_pids = set(feasible_pids)
                        pids.update(detected_pids)
                        for pid in detected_pids:
                            feasible_qids = set([q for q in self.reverse_dict[QID][pid] if q in self.entity_id_map and q in self.child_par_dict])
                            if types_in_context is None or QID_type_matched:
                                detected_qids = feasible_qids
                            else:
                                detected_qids = set([x for x in feasible_qids if len(set(self.child_all_par_dict[x]).intersection(types_in_context))>0])
                            if len(detected_qids)==0:
                                detected_qids = feasible_qids
                            for qid in detected_qids:#[q for q in self.wikidata[QID][pid] if q in self.entity_id_map]:
                                        tuples.add((QID, pid, qid))
                                        tuples.add((qid, pid, QID))
                    return tuples, pids

                try:
                    qn_entities = []
                    tagged_context_list = []
		    relations_in_context = set([])
                    types_in_context = set([])
                    for index,context in enumerate(dialogue_instance_context):
			if index==0:
				ques_entities, context_list = self.question_parser.get_utterance_entities(context, True)
			else:
	                        ques_entities, context_list = self.question_parser.get_utterance_entities(context)
			if create_vocab:
				self.word_counter.update([x.lower() for x in nltk.word_tokenize(context_list) if not self.isQid(x)])
			relation_in_context = self.find_relation(context, ques_entities)
                        relations_in_context.update(relation_in_context)
			type_in_context = self.find_type(context)
                        types_in_context.update(type_in_context)	
                        tagged_context_list.append(context_list)
			#if len(qn_entities)==0:
	                qn_entities.extend(ques_entities)
                    candidate_entities = qn_entities

                    if len(qn_entities) > MAX_CANDIDATE_ENTITIES:
                      candidate_entities = candidate_entities[:MAX_CANDIDATE_ENTITIES] #set(random.sample(qn_entities, MAX_CANDIDATE_ENTITIES))

                    dialogue_contexts.append(tagged_context_list)
		    dialogue_context_utterances.append(dialogue_instance_context)
		    response_entities, response_context_list = self.question_parser.get_utterance_entities(dialogue_instance_target)
		    words = [x.lower() for x in nltk.word_tokenize(response_context_list) if not self.isQid(x)]
		    if create_vocab: 
	                    self.word_counter.update(words)
        	            self.response_word_counter.update(words)
		    dialogue_responses.append(response_context_list)		
		    if len(relations_in_context)==0:
                        relations_in_context = None
                    if len(types_in_context)==0:
                        types_in_context = None				
                    tuples, relations_explored = get_tuples_involving_entities(candidate_entities, relations_in_context, types_in_context) # tuples are stored as (Qid,pid,Qid)
		    all_tuples = set(tuples)
		    '''
		    for hop in range(1):	
			    neighbour_entities = (extract_dimension_from_tuples_as_list(tuples, 2))
			    tuples = get_tuples_involving_entities(neighbour_entities)	     
		 	    all_tuples.update(tuples)
		    '''
                    if len(all_tuples) > MAX_CANDIDATE_TUPLES:
                      all_tuples = set(random.sample(all_tuples, MAX_CANDIDATE_TUPLES))
                    sources = extract_dimension_from_tuples_as_list(all_tuples, 0)
                    relations = extract_dimension_from_tuples_as_list(all_tuples, 1)
                    targets = extract_dimension_from_tuples_as_list(all_tuples, 2)
		    print 'len(sources,relations,targets) ',len(sources),len(relations),len(targets)
		    #sys.exit(1)
                    dialog_sources.append(get_str_of_seq(sources))
                    dialog_relations.append(get_str_of_seq(relations))
                    dialog_targets.append(get_str_of_seq(targets))
                    # print 'flag 4'

		
                except:
		    traceback.print_exc()
                    dialog_sources.append('')
                    dialog_relations.append('')
                    dialog_targets.append('')
                    sources = []
                    relations = []
                    targets = []

        with open(self.dialogue_context_file, 'a') as fp:
            for dialogue_instance in dialogue_contexts:
                dialogue_instance = '|'.join(dialogue_instance)
                fp.write(dialogue_instance+'\n')
        with open(self.dialogue_target_file, 'a') as fp:
            for dialogue_instance in dialogue_targets:
                fp.write(dialogue_instance+'\n')
	with open(self.dialogue_response_file,'a') as fp:
            for dialogue_instance in dialogue_responses:
                fp.write(dialogue_instance+'\n')
	with open(self.dialogue_orig_response_file,'a') as fp:
	    for dialogue_instance in dialogue_orig_responses:
		fp.write(dialogue_instance+'\n')
        with open(self.sources_file,'a') as fp:
            for source in dialog_sources:
                fp.write(source+'\n')
        with open(self.relation_file,'a') as fp:
            for rel in dialog_relations:
                fp.write(rel+'\n')
        with open(self.target_file,'a') as fp:
            for target in dialog_targets:
                fp.write(target+'\n')
        with open(self.active_set_file,'a') as fp:
            for active_set in dialog_active_sets:
                fp.write(active_set+'\n')


        
    def pad_or_clip_dialogue(self, dialogue_instance):
        if len(dialogue_instance)>(self.max_utter+1):
            return dialogue_instance[-(self.max_utter+1):]      
        elif len(dialogue_instance)<(self.max_utter+1):
            padded_dialogue_instance = []   
            pad_length = self.max_utter + 1 - len(dialogue_instance)
            padded_dialogue_instance = ['']*pad_length
            padded_dialogue_instance.extend(dialogue_instance)
            return padded_dialogue_instance
        else:
            return dialogue_instance

    def pad_or_clip_utterance(self, utterance):
        if len(utterance)>(self.max_len-2):
            utterance = utterance[:(self.max_len-2)]
            utterance.append(self.end_word_symbol)
            utterance.insert(0, self.start_word_symbol)
        elif len(utterance)<(self.max_len-2):
            pad_length = self.max_len - 2 - len(utterance)
            utterance.append(self.end_word_symbol)
            utterance.insert(0, self.start_word_symbol)
            utterance = utterance+[self.pad_symbol]*pad_length
        else:
            utterance.append(self.end_word_symbol)
            utterance.insert(0, self.start_word_symbol)
        return utterance

    def pad_or_clip_target(self, target_list):
        if len(target_list) > self.max_target_size:
            target_list = target_list[:self.max_target_size]
        elif len(target_list) < self.max_target_size:
            pad_length = self.max_target_size - len(target_list)
            target_list = target_list + ['<pad_kb>'] * pad_length
        return target_list

    def pad_or_clip_memory_entity(self, mem_list):
        if len(mem_list) > self.max_mem_size:
	    #print 'len(mem_list)', len(mem_list)
            #print 'self.max_mem_size', self.max_mem_size
            #raise Exception('len(mem_list) > self.max_mem_size')
            mem_list = mem_list[:self.max_mem_size]
        #elif len(mem_list) < self.max_mem_size:
        #    pad_length = self.max_mem_size - len(mem_list)
        #    mem_list = mem_list+['<pad_kb>']*pad_length
	mem_list = mem_list+['<nkb>']
        return mem_list

    def pad_or_clip_memory_rel(self, mem_list):
        if len(mem_list) > self.max_mem_size:
	    #print 'len(mem_list)', len(mem_list)
            #print 'self.max_mem_size', self.max_mem_size
            #raise Exception('len(mem_list) > self.max_mem_size')
            mem_list = mem_list[:self.max_mem_size]
        #elif len(mem_list) < self.max_mem_size:
        #    pad_length = self.max_mem_size - len(mem_list)
        #    mem_list = mem_list+['<pad_kb>']*pad_length
	mem_list = mem_list+['<nkb>']
        return mem_list

    def read_vocab(self):
        assert os.path.isfile(self.vocab_file)
        self.vocab_dict = {word:word_id for word_id, word in pkl.load(open(self.vocab_file, "r")).iteritems()}
	self.response_vocab_dict = {word:word_id for word_id, word in pkl.load(open(self.response_vocab_file, "r")).iteritems()}
        assert self.unk_symbol in self.vocab_dict
        assert self.start_word_symbol in self.vocab_dict
        assert self.end_word_symbol in self.vocab_dict
        assert self.pad_symbol in self.vocab_dict

    def build_vocab(self):
        # print 'vocab build started'
        total_freq = sum(self.word_counter.values())
        self.logger.info("Total word frequency in dictionary %d ", total_freq)

        if self.cutoff != -1:
            self.logger.info("Cutoff %d", self.cutoff)
            vocab_count = [x for x in self.word_counter.most_common() if x[1]>=self.cutoff]
	    response_vocab_count = [x for x in self.response_word_counter.most_common() if x[1]>=self.cutoff]
        else:
            vocab_count = [x for x in self.word_counter.most_common() if x[1]>=5]
	    response_vocab_count = [x for x in self.response_word_counter.most_common() if x[1]>=5]

        # print 'vocab count computed'
        self.safe_pickle(vocab_count, self.vocab_file.replace('.pkl','_counter.pkl'))
        self.vocab_dict = {self.unk_symbol:self.unknown_word_id, self.start_word_symbol:self.start_word_id, self.pad_symbol:self.pad_word_id, self.end_word_symbol:self.end_word_id, self.kb_word_symbol:self.kb_word_id}

	self.safe_pickle(response_vocab_count, self.response_vocab_file.replace('.pkl','_counter.pkl'))
	self.response_vocab_dict = {self.unk_symbol:self.unknown_word_id, self.start_word_symbol:self.start_word_id, self.pad_symbol:self.pad_word_id, self.end_word_symbol:self.end_word_id, self.kb_word_symbol:self.kb_word_id}

        i = 5
        for (word, count) in vocab_count:
            if not word in self.vocab_dict:
                self.vocab_dict[word] = i
                i += 1

	i = 5
	for (word, count) in response_vocab_count:
	    if not word in self.response_vocab_dict:
		self.response_vocab_dict[word] = i
		i += 1
        # print 'vocab dict formed'

        self.logger.info('Vocab size %d' % len(self.vocab_dict))
	self.logger.info('Response Vocab size %d' % len(self.response_vocab_dict))
   

    def binarize_corpus(self, dialogue_context_file, dialogue_target_file, dialogue_response_file, dialogue_orig_response_file, active_set_file, sources_file, relation_file, target_file, dialogue_pkl_file):
        binarized_corpus = []
        binarized_corpus_context = []
        binarized_corpus_target = []
        unknowns = 0.
        num_terms = 0.
        freqs = collections.defaultdict(lambda: 0)
        df = collections.defaultdict(lambda: 0)
        num_instances = 0
        with open(dialogue_context_file) as contextlines, open(dialogue_target_file) as targetlines, open(dialogue_response_file) as responselines, open(dialogue_orig_response_file) as orig_responselines, open(active_set_file) as active_set_lines, open(sources_file) as source_lines, open(relation_file) as relation_lines, open(target_file) as key_target_lines:
            for context, target, response, orig_response, active_set, source, relation, key_target in izip(contextlines, targetlines, responselines, orig_responselines, active_set_lines, source_lines, relation_lines, key_target_lines):
                context = context.lower().strip()
                num_instances += 1
                if num_instances % 1 == 0:
                    print 'finished ',num_instances, ' instances'
                utterances = context.split('|')
                binarized_context_1 = []
                binarized_context_2 = []

                for utterance in utterances:
                    try:
                        utterance_words = nltk.word_tokenize(utterance)
                    except:
                        utterance_words = utterance.split(' ')
                    utterance_words = self.pad_or_clip_utterance(utterance_words)
                    if self.end_word_symbol not in utterance_words:
                        print 'utterance ',utterance
                        print 'utterance words ',utterance_words
                        raise Exception('utterance does not have end symbol')
                    utterance_word_ids_1 = []
                    utterance_word_ids_2 = []

                    for word in utterance_words:
			if word in self.bad_qids:
                                word = self.wikidata_qid_to_name[word]
                        if not self.isQid(word):
                            word_id = self.vocab_dict.get(word, self.unknown_word_id)
			    freqs[word_id] += 1
                            utterance_word_ids_1.append(word_id)
                            if word_id != self.pad_word_id:
                                utterance_word_ids_2.append(self.kb_ov_idx) # last index + 1 of id to entity map
                            else:
                                utterance_word_ids_2.append(self.entity_id_map['<pad_kb>'])
                        else:
                            utterance_word_ids_1.append(self.kb_word_id)
                            utterance_word_ids_2.append(self.entity_id_map[word.upper()])
                        
                        unknowns += 1 * (word_id == self.unknown_word_id)
                    if self.end_word_id not in utterance_word_ids_1:
                        print 'utterance word ids ', utterance_word_ids_1
                        raise Exception('utterance word ids_1 does not have end word id')
                    num_terms += len(utterance_words)
                    binarized_context_1.append(utterance_word_ids_1)
                    binarized_context_2.append(utterance_word_ids_2)

                if len(binarized_context_1)!=self.max_utter:
                    raise Exception('binarized_text_context should be a list of length max_utter, found length ', len(binarized_context))
                target = target.rstrip()
                if len(target) > 0:
                    target = target.split('|')
                else:
                    target = []
		binarized_response = []
                response_length = 0
                try:
                        response_words = nltk.word_tokenize(response)
                except:
                        response_words = response.split(' ')
		response_words = self.pad_or_clip_utterance(response_words)
                for word in response_words:
			if word in self.bad_qids:
                                word = self.wikidata_qid_to_name[word]
                        if not self.isQid(word):
                                word_id = self.response_vocab_dict.get(word, self.unknown_word_id)
                                binarized_response.append(word_id)
                                freqs[word_id] += 1
                        else:
                                binarized_response.append(self.kb_word_id)
                response_length = binarized_response.index(self.end_word_id)
                source_word_ids = []
                relation_word_ids = []
                target_word_ids = []

                source = source.rstrip()
                relation = relation.rstrip()
                key_target = key_target.rstrip()

                if len(source) > 0:
                    source = source.split('|')
                else:
                    source = []

                if len(relation) > 0:
                    relation = relation.split('|')
                else:
                    relation = []

                if len(key_target) > 0:
                    key_target = key_target.split('|')
                else:
                    key_target = []
                source_words = self.pad_or_clip_memory_entity(source)
                relation_words = self.pad_or_clip_memory_rel(relation)
                key_target_words = self.pad_or_clip_memory_entity(key_target)

                for word in source_words:
                    word_id = self.entity_id_map[word]
                    source_word_ids.append(word_id)

                for word in relation_words:
                    word_id = self.rel_id_map[word]
                    relation_word_ids.append(word_id)

                for word in key_target_words:
                    word_id = self.entity_id_map[word]
                    target_word_ids.append(word_id)

                #active_set_list = active_set.rstrip().split('#')
		source_word_ids = "|".join([str(x) for x in source_word_ids])
                relation_word_ids = "|".join([str(x) for x in relation_word_ids])
                target_word_ids = "|".join([str(x) for x in target_word_ids])	
                binarized_corpus.append([binarized_context_1, binarized_context_2, target, binarized_response, response_length, orig_response, source_word_ids, relation_word_ids, target_word_ids, active_set])
        self.safe_pickle(binarized_corpus, dialogue_pkl_file)
        if not os.path.isfile(self.vocab_file):
            self.safe_pickle([(word, word_id, freqs[word_id], df[word_id]) for word, word_id in self.vocab_dict.items()], self.vocab_stats_file)
            inverted_vocab_dict = {word_id:word for word, word_id in self.vocab_dict.iteritems()}
            self.safe_pickle(inverted_vocab_dict, self.vocab_file)
            print 'dumped vocab in ', self.vocab_file
        self.logger.info("Number of unknowns %d" % unknowns)
        self.logger.info("Number of terms %d" % num_terms)
        self.logger.info("Mean document length %f" % float(sum(map(len, binarized_corpus))/len(binarized_corpus)))
        self.logger.info("Writing training %d dialogues (%d left out)" % (len(binarized_corpus), num_instances + 1 - len(binarized_corpus)))
        
            
