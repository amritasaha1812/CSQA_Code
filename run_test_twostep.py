#from annoy import AnnoyIndex
import math
import json
import sys
import os
sys.path.append(os.getcwd())
import math
import pickle as pkl
import random
import os.path
# import params
import numpy as np
from params_test import *
import nltk
from read_data import *
from hierarchy_model import Hierarchical_seq_model
import gensim
import unidecode
from collections import OrderedDict
import re


def feeding_dict(model, mem_size, inputs_w2v, inputs_kb, decoder_target, text_weights, mem_weights, decoder_inputs, sources, rel, key_target, feed_prev, ent_embedding, rel_embedding, batch_size):
    feed_dict = {}
    feed_dict[model.memory_size] = mem_size
    for encoder_input, input in zip(model.encoder_text_inputs_w2v, inputs_w2v):
        for encoder_input_i, input_i in zip(encoder_input, input):
            feed_dict[encoder_input_i] = input_i

    for encoder_input, input in zip(model.encoder_text_inputs_kb, inputs_kb):
        for encoder_input_i, input_i in zip(encoder_input, input):
            feed_dict[encoder_input_i] = input_i

    for encoder_input, input in zip(model.encoder_text_inputs_kb_emb, inputs_kb):
        for encoder_input_i, input_i in zip(encoder_input, input):
            feed_dict[encoder_input_i] = np.array([ent_embedding[i] for i in input_i], dtype=np.float32)

    for model_target_i in model.target_text:
        feed_dict[model_target_i] = np.zeros((batch_size), dtype=np.int32)

    for model_decoder_output_i, target_i in zip(model.decoder_text_outputs, decoder_target):
        feed_dict[model_decoder_output_i] = target_i

    for model_text_weight_i, weight_i in zip(model.text_weights, text_weights):
        feed_dict[model_text_weight_i] = weight_i
        feed_dict[model.mem_weights] = mem_weights

    for model_decoder_input_i, decoder_input_i in zip(model.decoder_text_inputs, decoder_inputs):
        feed_dict[model_decoder_input_i] = decoder_input_i
    feed_dict[model.sources_emb] = np.array([np.array([ent_embedding[i] for i in source]) for source in sources])
    feed_dict[model.rel_emb] = np.array([np.array([rel_embedding[i] for i in rel_i]) for rel_i in rel])	
    feed_dict[model.key_target_emb] = np.array([np.array([ent_embedding[i] for i in key_target_i]) for key_target_i in key_target])
    feed_dict[model.feed_previous] = feed_prev
    return feed_dict

def check_dir(param):
    if not os.path.exists(param['logs_path']):
        os.makedirs(param['logs_path'])
    if not os.path.exists(param['model_path']):
        os.makedirs(param['model_path'])
    if not os.path.exists(param['test_output_dir']):
	os.makedirs(param['test_output_dir']) 

def write_to_file(pred_op, true_op):
    pred_file = ''
    true_file = ''
    with open(true_file, 'w') as f_true:
        for true_sentence in true_op:
            f_true.write(true_sentence.strip()+'\n')

    with open(pred_file, 'w') as f_pred:
        for pred_sentence in pred_op:
            f_pred.write(pred_sentence.strip()+'\n')
    print 'Test (true and predicted) output written to corresponding files'

def run_testing(param):
    def get_test_op(model, batch_dict, ent_embedding, rel_embedding, id_entity_map, type_of_loss):
	if type_of_loss == "decoder":
                overriding_memory = 10
        else:
                overriding_memory = None
        memory_size, test_batch_enc_w2v, test_batch_enc_kb, batch_target, batch_orig_target, batch_response, batch_orig_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, batch_active_set = get_batch_data(param['max_len'], param['max_utter'], param['memory_size'], param['gold_target_size'], param['batch_size'], batch_dict, overriding_memory, is_test=True)
        feed_dict = feeding_dict(model, memory_size, test_batch_enc_w2v, test_batch_enc_kb, batch_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, True, ent_embedding, rel_embedding, param['batch_size'])
	if type_of_loss == "decoder":
		output_loss, output_prob = sess.run([losses, prob], feed_dict=feed_dict)
		return output_loss, output_prob, np.transpose(batch_response, (1,0)), batch_orig_response
	elif type_of_loss == "kvmem":
		output_loss, output_prob = sess.run([losses, prob], feed_dict=feed_dict)
		prob_mem_entries = output_prob
                prob_mem_entries = np.array(prob_mem_entries)
                mem_entries_sorted = np.fliplr(np.argsort(prob_mem_entries,axis=1))
                mem_attention_sorted = np.fliplr(np.sort(prob_mem_entries,axis=1))
                mem_entries_sorted = [[batch_key_target[j][i] for j in mem_entries_sorted[i]] for i in range(batch_key_target.shape[1])]
                mem_entries_sorted = np.array(mem_entries_sorted, dtype=np.int32)
                mem_entries_sorted = [[id_entity_map[mem_entries_sorted[i][j]] for j in range(mem_entries_sorted.shape[1])] for i in range(mem_entries_sorted.shape[0])]
                gold_entity_ids = batch_orig_target
                return output_loss, mem_entries_sorted, mem_attention_sorted, gold_entity_ids, batch_orig_response


    def perform_test(model, batch_dict, vocab, ent_embedding, rel_embedding, id_entity_map, type_of_loss, step, wikidata_id_name_map, wikidata_rel_id_name_map):
	if type_of_loss == "decoder":
		batch_test_loss, test_op, batch_target_word_ids, gold_orig_response = get_test_op(model, batch_dict, ent_embedding, rel_embedding, id_entity_map, type_of_loss)
		batch_predicted_sentence, prob_predicted_words, prob_true_words = get_predicted_sentence(test_op, batch_target_word_ids, vocab)
		print_pred_true_op_decoder(batch_predicted_sentence, gold_orig_response, batch_test_loss, step)
	elif type_of_loss == "kvmem":
		batch_test_loss, prob_mem_entries, prob_mem_scores, gold_entity_ids, gold_orig_response = get_test_op(model, batch_dict, ent_embedding, rel_embedding, id_entity_map, type_of_loss)
		print_pred_true_op_kvmem(prob_mem_entries, prob_mem_scores, gold_entity_ids, gold_orig_response, batch_test_loss, step, wikidata_id_name_map, wikidata_rel_id_name_map)
        sum_batch_loss = get_sum_batch_loss(batch_test_loss)
        sys.stdout.flush()
        return sum_batch_loss

    def get_predicted_sentence(test_op, true_op, vocab):
        max_probs_index = []
        max_probs = []
        if true_op is not None:
                true_op = true_op.tolist()
                true_op = np.asarray(true_op).T.tolist()
                true_op_prob = []
        i=0
        for op in test_op:
            sys.stdout.flush()
            max_index = np.argmax(op, axis=1)
            max_prob  = np.max(op, axis=1)
            max_probs.append(max_prob)
            max_probs_index.append(max_index)
            if true_op is not None:
                    true_op_prob.append([v_ij[t_ij] for v_ij,t_ij in zip(op, true_op[i])])
                    i=i+1
        max_probs_index = np.transpose(max_probs_index)
        max_probs = np.transpose(max_probs)
        if true_op is not None:
                true_op_prob = np.asarray(true_op_prob)
                true_op_prob = np.transpose(true_op_prob)
                if true_op_prob.shape[0]!=max_probs.shape[0] and true_op_prob.shape[1]!=max_probs.shape[1]:
                        raise Exception('some problem shape of true_op_prob' , true_op_prob.shape)
    #max_probs is of shape batch_size, max_len
        pred_sentence_list = map_id_to_word(max_probs_index, vocab)
        return pred_sentence_list, max_probs, true_op_prob

    def get_sum_batch_loss(batch_loss):
        return np.sum(np.asarray(batch_loss))
   
    def print_pred_true_op_decoder(pred_op, true_op, batch_test_loss, step):
	test_result_dir = param['test_output_dir']
	true_sent_file = os.path.join(test_result_dir, 'true_sent.txt')
	pred_sent_file = os.path.join(test_result_dir, 'pred_sent.txt')
	true_pred_sent_file = os.path.join(test_result_dir, 'true_pred_sent.txt')
	batch_loss_file = os.path.join(test_result_dir, 'decoder_loss.txt')
	if step == 0:
		mode = 'w'
	else:
		mode = 'a'
	f1 = open(true_sent_file, mode)
	f2 = open(pred_sent_file, mode)
	f3 = open(batch_loss_file, mode)
	f4 = open(true_pred_sent_file, mode)
	for i in range(0,len(true_op)):
            sys.stdout.flush()
            f1.write('%s\n' % true_op[i].strip())
            sys.stdout.flush()
            f2.write('%s\n' % pred_op[i].strip())
            print true_op[i].strip()+ '::::'+ pred_op[i].strip()
	    f3.write('%s\n' % str(batch_test_loss[i]))
	    f4.write('%s\n' % (true_op[i].strip()+'\t'+pred_op[i].strip()))
	f1.close()	
	f2.close()
	f3.close()
	f4.close()

    def print_pred_true_op_kvmem(prob_memory_entities, prob_memory_scores, gold_entity_ids, gold_orig_response, batch_test_loss, step, wikidata_id_name_map, wikidata_rel_id_name_map):
	test_result_dir = param['test_output_dir']
	top20_entid_from_mem_file = os.path.join(test_result_dir, 'top20_ent_id_from_mem.txt')
	top20_entid_from_kb_file = os.path.join(test_result_dir, 'top20_ent_id_from_kb.txt')
	top20_ent_from_mem_file = os.path.join(test_result_dir, 'top20_ent_from_mem.txt')
        top20_ent_from_kb_file = os.path.join(test_result_dir, 'top20_ent_from_kb.txt')
	gold_ent_file = os.path.join(test_result_dir, 'gold_ent.txt')
	gold_ent_id_file = os.path.join(test_result_dir, 'gold_ent_id.txt')
        gold_resp_file = os.path.join(test_result_dir, 'gold_resp.txt')
	batch_loss_file = os.path.join(test_result_dir, 'kvmem_loss.txt')
	if step==0:
                mode='w'
        else:
                mode='a'
	f1 = open(top20_entid_from_mem_file, mode)
	f2 = open(top20_entid_from_kb_file, mode)
	f3 = open(top20_ent_from_mem_file, mode)
	f4 = open(top20_ent_from_kb_file, mode)
	f5 = open(gold_ent_file, mode)
	f8 = open(gold_ent_id_file, mode)
	f6 = open(gold_resp_file, mode)
	f7 = open(batch_loss_file, mode)
	for i in range(0, len(batch_loss_file)):
	    print "top-5 memory entries from mem is:"
            memory_entities = [mem_entry for mem_entry in prob_memory_entities[i] if mem_entry not in ['<pad_kb>','<nkb>','<unk>']]
            f1.write('%s\n' % '|'.join([mem_entry for mem_entry in memory_entities][:20]))
            print ', '.join([mem_entry for mem_entry in memory_entities][:20])
            all_entities = []
            for entity in memory_entities[:10]:
                all_entities.append(entity)
                nbrs = []#ann.get_nns_by_item(entity,2)
                if nbrs is None:
                        raise Exception('neighbour set is none for ',entity)
                all_entities.extend(nbrs)
            f2.write('%s\n' % '|'.join([mem_entry for mem_entry in all_entities][:20]))
            print "top-5 KB entities in mem is:"
            memory_entities_qids = [id for id in memory_entities if id not in ['<pad_kb>','<nkb>','<unk>']]
            all_entities_qids = [id for id in all_entities if id not in ['<pad_kb>','<nkb>','<unk>']]
            all_entities_qids = list(OrderedDict.fromkeys(all_entities_qids))[:20]
            memory_entities_qids = list(OrderedDict.fromkeys(memory_entities_qids))[:20]
            f3.write('%s\n' % '|'.join([unidecode.unidecode(wikidata_id_name_map[id]) for id in memory_entities_qids if id in wikidata_id_name_map and wikidata_id_name_map[id] not in ['<pad_kb>','<nkb>','<unk>']][:20]))
            print ', '.join([unidecode.unidecode(wikidata_id_name_map[id]) for id in memory_entities_qids if id in wikidata_id_name_map and wikidata_id_name_map[id] not in ['<pad_kb>','<nkb>','<unk>']][:20])
            f4.write('%s\n' % '|'.join([unidecode.unidecode(wikidata_id_name_map[id]) for id in all_entities_qids if id in wikidata_id_name_map and wikidata_id_name_map[id] not in ['<pad_kb>','<nkb>','<unk>']][:20]))
	    print "gold entity ids in  is:"
	    f8.write('%s\n' % '|'.join([id for id in gold_entity_ids[i] if id in wikidata_id_name_map]))
            print ', '.join([id for id in gold_entity_ids[i] if id in wikidata_id_name_map])
            print "gold entities in  is:"
            f5.write('%s\n' % '|'.join([unidecode.unidecode(wikidata_id_name_map[id]) for id in gold_entity_ids[i] if id in wikidata_id_name_map]))
            print ', '.join([unidecode.unidecode(wikidata_id_name_map[id]) for id in gold_entity_ids[i] if id in wikidata_id_name_map])
            print "gold response in  is:"
            f6.write('%s\n' % gold_orig_response[i].strip())
            print gold_orig_response[i].strip()
            sys.stdout.flush()
            f7.write('%s\n' % str(batch_test_loss[i]))

        f1.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()
        f7.close()
	f8.close()
    
    def map_id_to_word(word_indices, vocab):
        sentence_list = []
        for sent in word_indices:
            word_list = []
            for word_index in sent:
                word = vocab[word_index]
                word_list.append(word)
            sentence_list.append(" ".join(word_list))
        return sentence_list
   
    wikidata_dir = param['wikidata_dir']
    transe_dir = param['transe_dir']
    glove_dir = param['glove_dir']
    wikidata_id_name_map=json.load(open(wikidata_dir+'/items_wikidata_n.json'))
    wikidata_rel_id_name_map=json.load(open(wikidata_dir+'/filtered_property_wikidata4.json')) 
    test_data = pkl.load(open(param['test_data_file']))
    print 'Test dialogue dataset loaded'
    sys.stdout.flush()
    vocab = pkl.load(open(param['vocab_file'],"rb"))
    vocab_size = len(vocab)
    response_vocab = pkl.load(open(param['response_vocab_file'],"rb"))
    response_vocab_size = len(response_vocab)	
    print 'test target sentence list loaded'
    check_dir(param)
    n_batches = int(math.ceil(float(len(test_data))/float(param['batch_size'])))
    print 'number of batches ', n_batches, 'len test data ', len(test_data), 'batch size' , param['batch_size']
    model_file = os.path.join(param['model_path'],"best_model")

    vocab_init_embed = np.empty([len(vocab.keys()), param['text_embedding_size']],dtype=np.float32)
    word2vec_pretrain_embed = gensim.models.Word2Vec.load_word2vec_format(glove_dir, binary=True)
    ent_embed = np.load(transe_dir+'/ent_embed.pkl.npy')
    rel_embed = np.load(transe_dir+'/rel_embed.pkl.npy')

    new_row = np.zeros((1,param['wikidata_embed_size']), dtype=np.float32)
    
    ent_embed = np.vstack([new_row, ent_embed]) # corr. to <pad_kb>
    ent_embed = np.vstack([new_row, ent_embed]) # corr. to <nkb>

    rel_embed = np.vstack([new_row, rel_embed]) # corr. to <pad_kb>
    rel_embed = np.vstack([new_row, rel_embed]) # corr. to <nkb>
    for i in xrange(vocab_init_embed.shape[0]):
        if vocab[i] in word2vec_pretrain_embed:
            vocab_init_embed[i,:] = word2vec_pretrain_embed[vocab[i]]
        elif i == 4: # KB word
	    vocab_init_embed[i,:] = np.zeros((1,vocab_init_embed.shape[1]),dtype=np.float32)
        else:
	    vocab_init_embed[i,:] = np.random.rand(1,vocab_init_embed.shape[1]).astype(np.float32)
    id_entity_map = {0:'<pad_kb>', 1: '<nkb>'}
    id_entity_map.update({(k+2):v for k,v in pkl.load(open(transe_dir+'/id_entity_map.pickle','rb')).iteritems()})

    id_rel_map = {0:'<pad_kb>', 1: '<nkb>'}
    id_rel_map.update({(k+2):v for k,v in pkl.load(open(transe_dir+'/id_rel_map.pickle','rb')).iteritems()})
    if 'type_of_loss' in param:
	type_of_loss = param['type_of_loss']
    else:
	type_of_loss = ''			
    with tf.Graph().as_default():
        model = Hierarchical_seq_model(param['text_embedding_size'], param['wikidata_embed_size'], param['cell_size'], param['cell_type'], param['batch_size'], param['learning_rate'], param['max_len'], param['max_utter'], param['patience'], param['max_gradient_norm'], param['activation'], param['output_activation'],vocab_init_embed, ent_embed, rel_embed, param['gold_target_size'], response_vocab_size, type_of_loss)
        model.create_placeholder()
        losses, prob = model.inference()
        print "model created"
        sys.stdout.flush()
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        saver.restore(sess, model_file)
        all_var = tf.all_variables()
        print 'printing all' , len(all_var),' TF variables:'
        for var in all_var:
            print var.name, var.get_shape()
        print 'testing started'
        sys.stdout.flush()
        test_loss = 0
        n_batches = int(math.ceil(float(len(test_data))/float(param['batch_size'])))
        for i in range(n_batches):
            batch_dict = test_data[i*param['batch_size']:(i+1)*param['batch_size']]
            sum_batch_loss = perform_test(model, batch_dict, response_vocab, ent_embed, rel_embed, id_entity_map, type_of_loss, i)
            test_loss = test_loss + sum_batch_loss

        print 'Avg. test loss = %f\n' % (float(test_loss)/float(len(test_data)))
        sys.stdout.flush()

        print 'Testing over'


def main():
    # test_type = sys.argv[2]
    param = get_params(sys.argv[1], sys.argv[2])
    # param = get_params(os.getcwd())
    print param
    if os.path.exists(param['test_data_file']):
        print 'dictionary already exists'
        sys.stdout.flush()
    else:
        get_dialog_dict_for_test(param, sys.argv[2])
        print 'dictionary formed'
        sys.stdout.flush()
    print param
    run_testing(param)

if __name__=="__main__":
    main()
