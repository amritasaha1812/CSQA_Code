import sys
import numpy as np
import cPickle as pkl
import os
#from params import *
from prepare_data_for_hred import PrepareData as PrepareData
start_symbol_index = 0
end_symbol_index = 1
unk_symbol_index = 2
pad_symbol_index = 3
kb_pad_idx = 0
nkb = 1
import os


def get_dialog_dict(param):
    train_dir_loc = param['train_dir_loc']
    valid_dir_loc = param['valid_dir_loc']
    test_dir_loc = param['test_dir_loc']
    dump_dir_loc = param['dump_dir_loc']
    vocab_file = param['vocab_file']
    vocab_stats_file = param['vocab_stats_file']
    vocab_freq_cutoff = param['vocab_freq_cutoff']
    train_data_file = param['train_data_file']
    valid_data_file = param['valid_data_file']
    test_data_file = param['test_data_file']
    max_utter = param['max_utter']
    max_len = param['max_len']
    stopwords = param['stopwords']
    stopwords_histogram = param['stopwords_histogram']
    max_mem_size = param['memory_size']
    max_target_size = param['gold_target_size']
    ques_type_id = param['ques_type_id']
    ques_type_name = param['ques_type_name']
    vocab_max_len = param['vocab_max_len']
    wikidata_dir = param['wikidata_dir']
    lucene_dir = param['lucene_dir'] 
    transe_dir = param['transe_dir']
    glove_dir = param['glove_dir']
    preparedata = PrepareData(max_utter, max_len, start_symbol_index, end_symbol_index, unk_symbol_index, pad_symbol_index, kb_pad_idx, nkb, stopwords, stopwords_histogram, lucene_dir, transe_dir, wikidata_dir, glove_dir, max_mem_size, max_target_size, vocab_max_len, True, cutoff=vocab_freq_cutoff)
    if os.path.isfile(vocab_file):
        print 'found existing vocab file in '+str(vocab_file)+', ... reading from there'
    print 'to delete later ',os.path.join(dump_dir_loc, "train")	
    preparedata.prepare_data(train_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "train"), train_data_file, ques_type_id, ques_type_name)
    preparedata.prepare_data(valid_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "valid"), valid_data_file, ques_type_id, ques_type_name)
    #preparedata.prepare_data(test_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "test"), test_data_file, ques_type_id)


def get_dialog_dict_for_test(param):
    test_dir_loc = param['test_dir_loc']
    dump_dir_loc = param['dump_dir_loc']
    vocab_file = param['vocab_file']
    vocab_stats_file = param['vocab_stats_file']
    response_vocab_file = param['response_vocab_file']
    vocab_freq_cutoff = param['vocab_freq_cutoff']
    test_data_file = param['test_data_file']
    max_utter = param['max_utter']
    max_len = param['max_len']
    stopwords = param['stopwords']
    stopwords_histogram = param['stopwords_histogram']
    max_mem_size = param['memory_size']
    max_target_size = param['gold_target_size']
    ques_type_id = param['ques_type_id']
    ques_type_name = param['ques_type_name']
    vocab_max_len = param['vocab_max_len']
    wikidata_dir = param['wikidata_dir']
    lucene_dir = param['lucene_dir']
    transe_dir = param['transe_dir']
    glove_dir = param['glove_dir'] 
    preparedata = PrepareData(max_utter, max_len, start_symbol_index, end_symbol_index, unk_symbol_index, pad_symbol_index, kb_pad_idx, nkb, stopwords, stopwords_histogram, lucene_dir, transe_dir, wikidata_dir, glove_dir, max_mem_size, max_target_size, vocab_max_len, True, cutoff=vocab_freq_cutoff)
    if os.path.isfile(vocab_file):
        print 'found existing vocab file in '+str(vocab_file)+', ... reading from there'
    print 'to delete later ',os.path.join(dump_dir_loc, "train")
    preparedata.prepare_data(test_dir_loc, vocab_file, vocab_stats_file, response_vocab_file, os.path.join(dump_dir_loc, "test"), test_data_file, ques_type_id, ques_type_name)



def get_utter_seq_len(dialogue_dict_w2v, dialogue_dict_kb, dialogue_target, dialogue_response, dialogue_response_length, dialogue_sources, dialogue_rel, dialogue_key_target, max_len, max_utter, max_target_size, max_mem_size, batch_size, is_test=False):
    padded_utters_id_w2v = None
    padded_utters_id_kb = None
    padded_target =[]
    decode_seq_len = []
    padded_utters_id_w2v = np.asarray([[xij for xij in dialogue_i] for dialogue_i in dialogue_dict_w2v])
    padded_utters_id_kb = np.asarray([[xij for xij in dialogue_i] for dialogue_i in dialogue_dict_kb])
    if not is_test:	
    	padded_target = np.asarray([xi for xi in dialogue_target])
    else:
	padded_target = dialogue_target
    padded_response = np.asarray([xi for xi in dialogue_response])
    pad_to_response = np.reshape(np.array([pad_symbol_index]*batch_size), (batch_size, 1))
    padded_decoder_input = np.concatenate((pad_to_response, padded_response[:,:-1]), axis=1)
    padded_response_length = np.asarray(dialogue_response_length)
    padded_sources = np.asarray([xi[:-1-max(0, len(xi)-max_mem_size)]+[kb_pad_idx]*(max(0,max_mem_size-len(xi)))+[xi[-1]] for xi in dialogue_sources], dtype=np.int32) 
    padded_rel = np.asarray([xi[:-1-max(0, len(xi)-max_mem_size)]+[kb_pad_idx]*(max(0,max_mem_size-len(xi)))+[xi[-1]] for xi in dialogue_rel],  dtype=np.int32)
    padded_key_target = np.asarray([xi[:-1-max(0, len(xi)-max_mem_size)]+[kb_pad_idx]*(max(0,max_mem_size-len(xi)))+[xi[-1]] for xi in dialogue_key_target],  dtype=np.int32)
    return padded_utters_id_w2v, padded_utters_id_kb, padded_target, padded_response, padded_response_length, padded_decoder_input, padded_sources, padded_rel, padded_key_target

def get_weights(batch_size, max_len, actual_len):
    remaining_len = max_len - actual_len
    weights = [[1.]*actual_len_i+[0.]*remaining_len_i for actual_len_i,remaining_len_i in zip(actual_len,remaining_len)]
    weights = np.asarray(weights)
    return weights

def get_memory_weights(batch_size, max_mem_size, sources, rel, target):
    weights = np.ones((batch_size, max_mem_size))
    weights[np.where(sources==kb_pad_idx)] = 0.
    weights[np.where(rel==kb_pad_idx)] = 0.
    weights[np.where(target==kb_pad_idx)] = 0.
    weights[np.where(sources==nkb)] = 0.
    weights[np.where(rel==nkb)] = 0.
    weights[np.where(target==nkb)] = 0.
    return weights

def get_batch_data(max_len, max_utter, max_mem_size, max_target_size, batch_size, data_dict, overriding_memory=None, is_test=False):
    data_dict = np.asarray(data_dict)
    batch_enc_w2v = data_dict[:,0]
    batch_enc_kb = data_dict[:,1]
    batch_target = data_dict[:,2]
    batch_response = data_dict[:,3]
    batch_response_length = data_dict[:,4]
    batch_orig_response = data_dict[:,5]
    batch_sources = [x.split("|") for x in data_dict[:,6]]
    batch_rel = [x.split("|") for x in data_dict[:,7]]
    batch_key_target = [x.split("|") for x in data_dict[:,8]]
    if is_test:
	batch_orig_response_entities = data_dict[:,10]
    else:
	batch_orig_response_entities = ['']*data_dict.shape[0]
    if isinstance(batch_orig_response_entities, np.ndarray):
	batch_orig_response_entities = batch_orig_response_entities.tolist()	
    if len(batch_orig_response_entities)!=batch_size:
	batch_orig_response_entities.extend(['']*(batch_size-len(batch_orig_response_entities)))
    if overriding_memory is not None:
	batch_sources = [x[:-1][:overriding_memory-1]+[x[-1]] for x in batch_sources]
	batch_rel = [x[:-1][:overriding_memory-1]+[x[-1]] for x in batch_rel]
	batch_key_target = [x[:-1][:overriding_memory-1]+[x[-1]] for x in batch_key_target]
    '''
    try:
	    batch_active_set = data_dict[:,9]
    except:
	    batch_active_set = ['']*len(data_dict)
    '''
    batch_active_set = ['']*len(data_dict)
    batch_response_len = [len(response) for response in batch_response]
    orig_lens = [len(batch_sources_i) for batch_sources_i in batch_sources]
    max_mem_size = max(orig_lens)
    avg_mem_size = float(sum(orig_lens))/float(len(batch_sources))
    if max_mem_size - avg_mem_size >500:
        print 'WARNING: max_mem_size ',max_mem_size, 'avg_mem_size ',avg_mem_size
    if len(data_dict) % batch_size != 0:
	   batch_enc_w2v, batch_enc_kb, batch_target, batch_response, batch_response_length, batch_orig_response, batch_sources, batch_rel, batch_key_target, batch_active_set = check_padding(batch_enc_w2v, batch_enc_kb, batch_target, batch_response, batch_response_length, batch_orig_response, batch_sources, batch_rel, batch_key_target, batch_active_set, max_len, max_utter, max_mem_size, max_target_size, batch_size, is_test)
    
    padded_enc_w2v, padded_enc_kb, padded_target, padded_response, padded_response_length, padded_decoder_input, padded_batch_sources, padded_batch_rel, padded_batch_key_target = get_utter_seq_len(batch_enc_w2v, batch_enc_kb, batch_target, batch_response, batch_response_length, batch_sources, batch_rel, batch_key_target, max_len, max_utter, max_target_size, max_mem_size, batch_size, is_test)
    
    padded_weights = get_weights(batch_size, max_len, padded_response_length)
    padded_memory_weights = get_memory_weights(batch_size, max_mem_size, padded_batch_sources, padded_batch_rel, padded_batch_key_target)
    
    padded_enc_w2v, padded_enc_kb, padded_target, padded_orig_target, padded_response, padded_weights, padded_decoder_input, padded_batch_sources, padded_batch_rel, padded_batch_key_target = transpose_utterances(padded_enc_w2v, padded_enc_kb, padded_target, padded_response, padded_weights, padded_decoder_input, padded_batch_sources, padded_batch_rel, padded_batch_key_target, max_mem_size, batch_size, is_test)
       
    return max_mem_size, padded_enc_w2v, padded_enc_kb, padded_target, padded_orig_target, padded_response, batch_orig_response, padded_weights, padded_memory_weights, padded_decoder_input, padded_batch_sources, padded_batch_rel, padded_batch_key_target, batch_active_set, batch_orig_response_entities

def transpose_utterances(padded_enc_w2v, padded_enc_kb, padded_target, padded_response, padded_weights, padded_decoder_input, batch_sources, batch_rel, batch_key_target, max_mem_size, batch_size, is_test):

    batch_key_target = np.asarray(batch_key_target) # batch_size * max_mem_size
    # padded_target : batch_size * max_target_size
    if not is_test:
	    #print 'padded_target shape ', padded_target.shape
	    mapped_padded_target = np.zeros(padded_target.shape)
	    for i in xrange(mapped_padded_target.shape[0]):
        	for j in xrange(mapped_padded_target.shape[1]):
	            if padded_target[i,j] in batch_key_target[i,:] and padded_target[i,j] != kb_pad_idx:
        	        mapped_padded_target[i,j] = np.nonzero(batch_key_target[i,:] == padded_target[i,j])[0][0]
	            elif padded_target[i,j] not in batch_key_target[i,:]:
        	        mapped_padded_target[i,j] = max_mem_size-1 
	            else:
        	        mapped_padded_target[i,j] = max_mem_size-1 
    padded_transposed_enc_w2v = padded_enc_w2v.transpose((1,2,0))
    padded_transposed_enc_kb = padded_enc_kb.transpose((1,2,0))
    if not is_test:
    	padded_transposed_target = mapped_padded_target.transpose((1,0))
    else:
	padded_transposed_target = padded_target
    padded_transposed_response = padded_response.transpose((1,0))
    padded_transposed_weights = padded_weights.transpose((1,0))
    padded_transposed_decoder_input = padded_decoder_input.transpose((1,0))
    if not is_test:
    	padded_transposed_orig_target = padded_target.transpose((1,0))
    else:
	padded_transposed_orig_target = padded_target
    padded_batch_sources = np.asarray(batch_sources).transpose((1,0))
    padded_batch_rel = np.asarray(batch_rel).transpose((1,0))
    padded_batch_key_target = np.asarray(batch_key_target).transpose((1,0))

    return padded_transposed_enc_w2v, padded_transposed_enc_kb, padded_transposed_target, padded_transposed_orig_target, padded_transposed_response, padded_transposed_weights, padded_transposed_decoder_input, padded_batch_sources, padded_batch_rel, padded_batch_key_target

def batch_padding_context(data_mat, max_len, max_utter, pad_size):
    empty_data = [start_symbol_index, end_symbol_index]+[pad_symbol_index]*(max_len-2)
    empty_data = [empty_data]*max_utter
    empty_data_mat = [empty_data]*pad_size
    data_mat=data_mat.tolist()
    data_mat.extend(empty_data_mat)
    return data_mat

def batch_padding_target(data_mat, max_target_size, pad_size, is_test=False):
    if not is_test:
	    empty_data = [kb_pad_idx] * max_target_size
	    empty_data = [empty_data] * pad_size
	    data_mat=data_mat.tolist()
	    data_mat.extend(empty_data)
    else:
	if isinstance(data_mat, list):
		data_mat.extend(['']*pad_size)
	else:
		data_mat=data_mat.tolist()
		data_mat.extend(['']*pad_size)
    return data_mat

def batch_padding_response(data_mat, max_len, pad_size):
    empty_data = [start_symbol_index, end_symbol_index]+[pad_symbol_index]*(max_len-2)
    empty_data_mat = [empty_data]*pad_size
    data_mat=data_mat.tolist()
    data_mat.extend(empty_data_mat)
    return data_mat

def batch_padding_response_len(data_mat, pad_size):
    empty_data_mat = [2]*pad_size
    data_mat = data_mat.tolist()
    data_mat.extend(empty_data_mat)
    return data_mat

def batch_padding_orig_response(data_mat, pad_size):
    data_mat = data_mat.tolist()
    data_mat.extend(['']*pad_size)
    return data_mat

def batch_padding_active_set(data_mat, pad_size):
    if not isinstance(data_mat, list):
	data_mat = data_mat.tolist()
    data_mat.extend(['']*pad_size)
    return data_mat

def batch_padding_memory_ent(data_mat, max_mem_size, pad_size):
    empty_data = [kb_pad_idx]*(max_mem_size)
    empty_data = [empty_data]*pad_size
    if not isinstance(data_mat, list):
	data_mat=data_mat.tolist()
    data_mat.extend(empty_data)
    return data_mat

def batch_padding_memory_rel(data_mat, max_mem_size, pad_size):
    empty_data = [kb_pad_idx]*(max_mem_size)
    empty_data = [empty_data]*pad_size
    if not isinstance(data_mat, list):	
	data_mat=data_mat.tolist()
    data_mat.extend(empty_data)
    return data_mat

def check_padding(batch_enc_w2v, batch_enc_kb, batch_target, batch_response, batch_response_length, batch_orig_response, batch_sources, batch_rel, batch_key_target, batch_active_set, max_len, max_utter, max_mem_size, max_target_size, batch_size, is_test=False):
    pad_size = batch_size - len(batch_target) % batch_size
    batch_enc_w2v = batch_padding_context(batch_enc_w2v, max_len, max_utter, pad_size)
    batch_enc_kb = batch_padding_context(batch_enc_kb, max_len, max_utter, pad_size)
    batch_target = batch_padding_target(batch_target, max_target_size, pad_size, is_test)
    batch_response = batch_padding_response(batch_response, max_len, pad_size)
    batch_response_length = batch_padding_response_len(batch_response_length, pad_size)
    batch_orig_response = batch_padding_orig_response(batch_orig_response, pad_size)
    batch_sources = batch_padding_memory_ent(batch_sources, max_mem_size, pad_size) # adding one dummy entry for OOM entities
    batch_rel = batch_padding_memory_rel(batch_rel, max_mem_size, pad_size) # adding one dummy entry for OOM entities
    batch_key_target = batch_padding_memory_ent(batch_key_target, max_mem_size, pad_size) # adding one dummy entry for OOM entities
    batch_active_set = batch_padding_active_set(batch_active_set, pad_size)
    return batch_enc_w2v, batch_enc_kb, batch_target, batch_response, batch_response_length, batch_orig_response, batch_sources, batch_rel, batch_key_target, batch_active_set


def load_valid_test_target(data_dict):
    return np.asarray(data_dict)[:,3]

