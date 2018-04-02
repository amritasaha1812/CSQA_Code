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
from hierarchy_model import *
import gensim
import unidecode
from collections import OrderedDict
import re


def feeding_dict(model, inputs_w2v, inputs_kb, orig_target, target, decoder_target, text_weights, mem_weights, decoder_inputs, sources, rel, key_target, feed_prev, ent_embedding):
    feed_dict = {}
    for encoder_input, input in zip(model.encoder_text_inputs_w2v, inputs_w2v):
        for encoder_input_i, input_i in zip(encoder_input, input):
            feed_dict[encoder_input_i] = input_i

    for encoder_input, input in zip(model.encoder_text_inputs_kb, inputs_kb):
        for encoder_input_i, input_i in zip(encoder_input, input):
            feed_dict[encoder_input_i] = input_i

    for encoder_input, input in zip(model.encoder_text_inputs_kb_emb, inputs_kb):
        for encoder_input_i, input_i in zip(encoder_input, input):
            feed_dict[encoder_input_i] = np.array([ent_embedding[i] for i in input_i], dtype=np.float32)

    for model_target_i, target_i in zip(model.target_text, target):
        feed_dict[model_target_i] = target_i

    for model_decoder_output_i, target_i in zip(model.decoder_text_outputs, decoder_target):
        feed_dict[model_decoder_output_i] = target_i

    for model_text_weight_i, weight_i in zip(model.text_weights, text_weights):
        feed_dict[model_text_weight_i] = weight_i
        feed_dict[model.mem_weights] = mem_weights

    for model_decoder_input_i, decoder_input_i in zip(model.decoder_text_inputs, decoder_inputs):
        feed_dict[model_decoder_input_i] = decoder_input_i

    for model_source, source in zip(model.sources, sources):
        feed_dict[model_source] = source

    for model_source_emb, source in zip(model.sources_emb, sources):
        feed_dict[model_source_emb] = np.array([ent_embedding[i] for i in source], dtype=np.float32)
    for model_rel, relation in zip(model.rel, rel):
        feed_dict[model_rel] = relation
    for model_key_target, key_target_i in zip(model.key_target, key_target):
        feed_dict[model_key_target] = key_target_i

    for model_key_target, key_target_i in zip(model.key_target_emb, key_target):
        feed_dict[model_key_target] = np.array([ent_embedding[i] for i in key_target_i], dtype=np.float32)
 
    for model_gold_emb, orig_target_i in zip(model.gold_emb, orig_target):
        feed_dict[model_gold_emb] = np.array([ent_embedding[i] for i in orig_target_i], dtype=np.float32)
 	
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
    def get_test_op(model, batch_dict, ent_embedding, id_entity_map):
        test_batch_enc_w2v, test_batch_enc_kb, batch_target, batch_orig_target, batch_response, batch_orig_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, batch_active_set = get_batch_data(param['max_len'], param['max_utter'], param['memory_size'], param['gold_target_size'], param['batch_size'], batch_dict)
        feed_dict = feeding_dict(model, test_batch_enc_w2v, test_batch_enc_kb, batch_orig_target, batch_target, batch_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, True, ent_embedding)
        dec_op, loss, prob_mem_entries = sess.run([logits, losses, prob_mem], feed_dict=feed_dict)
        prob_mem_entries = np.array(prob_mem_entries)
	batch_target = np.transpose(batch_target, (1,0)) # batch_size * target_size
	batch_target_in_mem = [[(batch_key_target[int(id)][i], prob_mem_entries[i][int(id)]) for id in batch_target[i] if id!=param['memory_size']] for i in range(batch_target.shape[0])]
	mem_entries_values_sorted = np.fliplr(np.sort(prob_mem_entries,axis=1))
	#print 'memory sorted ',[mem_entries_values_sorted[i][:10] for i in range(mem_entries_values_sorted.shape[0])]
        mem_entries_sorted = np.fliplr(np.argsort(prob_mem_entries,axis=1))
	#print 'memory_sorted_index ',[mem_entries_sorted[i][:10] for i in range(mem_entries_sorted.shape[0])]
        mem_entries_sorted_idx = mem_entries_sorted
	
        mem_entries_sorted = [[batch_key_target[j][i] for j in mem_entries_sorted[i]] for i in range(batch_key_target.shape[1])]

        mem_tuples_sorted = [[[batch_sources[j][i], batch_rel[j][i], batch_key_target[j][i]] for j in mem_entries_sorted_idx[i]] for i in range(batch_key_target.shape[1])]
        mem_tuples_sorted = np.asarray(mem_tuples_sorted, dtype=np.int32)

        mem_entries_sorted = np.array(mem_entries_sorted, dtype=np.int32)
        #mem_entries_sorted = [[id_entity_map[mem_entries_sorted[i][j]] for j in range(mem_entries_sorted.shape[1])] for i in range(mem_entries_sorted.shape[0])]

        mem_tuples_sorted = [[[id_entity_map[mem_tuples_sorted[i][j][0]], id_rel_map[mem_tuples_sorted[i][j][1]], id_entity_map[mem_tuples_sorted[i][j][2]]]  for j in range(mem_tuples_sorted.shape[1])] for i in range(mem_tuples_sorted.shape[0])]

        gold_entity_ids = np.transpose(batch_orig_target, (1,0))
        gold_entity_ids = [[id_entity_map[gold_entity_ids[i][j]] for j in range(gold_entity_ids.shape[1])] for i in range(gold_entity_ids.shape[0])]
        return loss, dec_op, mem_entries_sorted, gold_entity_ids, batch_orig_response, mem_tuples_sorted, batch_target_in_mem, batch_active_set

    def perform_test(model, batch_dict, batch_target_word_ids, batch_text_targets, vocab, ent_embedding, id_entity_map, batch_count, wikidata_id_name_map, wikidata_rel_id_name_map):
        batch_test_loss, test_op, prob_mem_entries, gold_entity_ids, gold_orig_response, prob_mem_tuples, batch_target_in_mem, batch_active_set  = get_test_op(model, batch_dict, ent_embedding, id_entity_map)
        sum_batch_loss = get_sum_batch_loss(batch_test_loss)
        batch_predicted_sentence, prob_predicted_words, prob_true_words = get_predicted_sentence(test_op, batch_target_word_ids, vocab)
        sys.stdout.flush()
    
        print_pred_true_op(batch_predicted_sentence, prob_predicted_words, prob_true_words, batch_text_targets, batch_test_loss, prob_mem_entries, gold_entity_ids, gold_orig_response, prob_mem_tuples, batch_count, batch_target_in_mem, id_entity_map, batch_active_set, wikidata_id_name_map, wikidata_rel_id_name_map)
        sys.stdout.flush()
        return sum_batch_loss

    def get_sum_batch_loss(batch_loss):
        return np.sum(np.asarray(batch_loss))

    def replace_kb_ent_in_resp(prob_memory_entities, pred_op, id_entity_map, batch_target_in_mem):
	batch_target_in_mem_names = [unidecode.unidecode(wikidata_id_name_map[id_entity_map[id[0]]]) for id in batch_target_in_mem]
	prob_memory_entities_qids = [id_entity_map[id] for id in prob_memory_entities]	
	kb_name_list = [unidecode.unidecode(wikidata_id_name_map[id]) for id in prob_memory_entities_qids if id in wikidata_id_name_map if wikidata_id_name_map[id] not in ['<pad_kb>','<nkb>','<unk>']]
        kb_name_list_unique = list(OrderedDict.fromkeys(kb_name_list))[:20]
        pred_op_new = pred_op.split(' ')
        k = 0
        max_k = 5
        length = len(pred_op_new)
        replace_kb = True
        for j in range(len(pred_op_new)):
                if pred_op_new[j] in ['<pad_kb>','<nkb>','<unk>','</s>','<pad>']:
                        pred_op_new[j] = ''
                if pred_op_new[j]=='</e>':
                        length = j
                if pred_op_new[j] == '<kb>':
                    if not replace_kb:
                        pred_op_new[j] = ''
                        continue
                    if k == len(kb_name_list_unique) or k == max_k:
                        replace_kb = False
                        pred_op_new[j] = ''
                        continue
                    pred_op_new[j] = kb_name_list_unique[k]
                    k = k+1
        pred_op_new = pred_op_new[:length]
        pred_op_new = re.sub(' +',' ',' '.join(pred_op_new)).strip()
	return pred_op_new  

    def print_pred_true_op(pred_op, prob_pred, prob_true, true_op, batch_valid_loss, prob_memory_entities, gold_entity_ids, gold_orig_response, prob_mem_tuples, batch_count, batch_target_in_mem, id_entity_map, batch_active_set, wikidata_id_name_map, wikidata_rel_id_name_map):
	test_result_dir = param['test_output_dir']
        true_sent_file = os.path.join(test_result_dir, 'true_sent.txt')
	pred_sent_file = os.path.join(test_result_dir, 'pred_sent.txt')
        pred_sent_mem_file = os.path.join(test_result_dir, 'pred_sent_replace_ent_from_mem.txt')
	pred_sent_kb_file = os.path.join(test_result_dir, 'pred_sent_replace_ent_from_kb.txt')
	top20_entid_from_mem_file = os.path.join(test_result_dir, 'top20_ent_id_from_mem.txt')
	top20_entid_from_kb_file = os.path.join(test_result_dir, 'top20_ent_id_from_kb.txt')
	top20_ent_from_mem_file = os.path.join(test_result_dir, 'top20_ent_from_mem.txt')
        top20_ent_from_kb_file = os.path.join(test_result_dir, 'top20_ent_from_kb.txt')
        gold_ent_file = os.path.join(test_result_dir, 'gold_ent.txt')
        gold_resp_file = os.path.join(test_result_dir, 'gold_resp.txt')
        batch_loss_file = os.path.join(test_result_dir, 'loss.txt')
        top20_mem_tuple_file = os.path.join(test_result_dir, 'top20_mem_tuple.txt')
	gold_ent_in_mem_file = os.path.join(test_result_dir, 'gold_ent_in_mem.txt')
	gold_ent_in_mem_not_in_top_file = os.path.join(test_result_dir, 'gold_ent_in_mem_not_in_top.txt')
	active_set_file = os.path.join(test_result_dir, 'active_set.txt')
	if batch_count==0:
		mode='w'
	else:
		mode='a'
        f1 = open(true_sent_file, mode)
	f15 = open(pred_sent_file, mode)
        f2 = open(pred_sent_mem_file, mode)
	f3 = open(pred_sent_kb_file, mode)
	f4 = open(top20_entid_from_mem_file, mode)
	f5 = open(top20_entid_from_kb_file, mode)
        f6 = open(top20_ent_from_mem_file, mode)
	f7 = open(top20_ent_from_kb_file, mode)
        f8 = open(gold_ent_file, mode)
        f9 = open(gold_resp_file, mode)
        f10 = open(batch_loss_file, mode)
        f11 = open(top20_mem_tuple_file, mode)
	f12 = open(gold_ent_in_mem_file, mode)
	f13 = open(gold_ent_in_mem_not_in_top_file, mode)
	f14 = open(active_set_file, mode)
        for i in range(0,len(true_op)):
            #print "true sentence in  is:"
            #sys.stdout.flush()
            f1.write('%s\n' % true_op[i])
	    #print true_op[i]
	    if i%20==0:
	            print "predicted sentence in:"+pred_op[i] +' ('+true_op[i]+')'
        	    sys.stdout.flush()
	    f15.write('%s\n' % pred_op[i])
    	    #print "top-5 memory entries from mem is:"
	    memory_entities = [mem_entry for mem_entry in prob_memory_entities[i] if mem_entry not in ['<pad_kb>','<nkb>','<unk>']]
       	    f4.write('%s\n' % ', '.join([unidecode.unidecode(id_entity_map[mem_entry]) for mem_entry in memory_entities][:20]))
	    #print ', '.join([unidecode.unidecode(id_entity_map[mem_entry]) for mem_entry in memory_entities][:20])
	    all_entities = []
	    for entity in memory_entities[:10]:
		all_entities.append(entity)
	    f5.write('%s\n' % ', '.join([unidecode.unidecode(id_entity_map[mem_entry]) for mem_entry in all_entities][:20]))	    	
	    memory_entities_qids = [id_entity_map[id] for id in memory_entities if id_entity_map[id] not in ['<pad_kb>','<nkb>','<unk>']]
	    all_entities_qids = [id_entity_map[id] for id in all_entities if id_entity_map[id] not in ['<pad_kb>','<nkb>','<unk>']]
	    all_entities_qids = list(OrderedDict.fromkeys(all_entities_qids))[:20]
            memory_entities_qids = list(OrderedDict.fromkeys(memory_entities_qids))[:20]
    	    f6.write('%s\n' % ', '.join([unidecode.unidecode(wikidata_id_name_map[id]) for id in memory_entities_qids if id in wikidata_id_name_map and wikidata_id_name_map[id] not in ['<pad_kb>','<nkb>','<unk>']][:20]))
	    f7.write('%s\n' % ', '.join([unidecode.unidecode(wikidata_id_name_map[id]) for id in all_entities_qids if id in wikidata_id_name_map and wikidata_id_name_map[id] not in ['<pad_kb>','<nkb>','<unk>']][:20]))
	   
	    pred_op_replaced_ent_from_mem = replace_kb_ent_in_resp(prob_memory_entities[i], pred_op[i], id_entity_map, batch_target_in_mem[i])
	    f2.write('%s\n' % pred_op_replaced_ent_from_mem)
	    pred_op_replaced_ent_from_kb = replace_kb_ent_in_resp(all_entities, pred_op[i], id_entity_map, batch_target_in_mem[i])
	    f3.write('%s\n' % pred_op_replaced_ent_from_kb)
    	    f8.write('%s\n' % ', '.join([unidecode.unidecode(wikidata_id_name_map[id]) for id in gold_entity_ids[i] if id in wikidata_id_name_map]))
            f9.write('%s\n' % gold_orig_response[i].strip())
            f10.write('%s\n' % str(batch_valid_loss[i]))
	    batch_target_in_mem_str = [unidecode.unidecode(wikidata_id_name_map[id_entity_map[id[0]]]) for id in batch_target_in_mem[i] if id_entity_map[id[0]] not in ['<pad_kb>','<nkb>','<unk>']]
	    batch_target_in_mem_not_in_top_str = [unidecode.unidecode(wikidata_id_name_map[id_entity_map[id[0]]]+'('+str(id[1])+')') for id in batch_target_in_mem[i] if id_entity_map[id[0]] not in ['<pad_kb>','<nkb>','<unk>'] and id_entity_map[id[0]] not in all_entities_qids and id_entity_map[id[0]] not in memory_entities_qids]
	    f12.write('%s\n' % ', '.join(batch_target_in_mem_str))
	    f13.write('%s\n' % ', '.join(batch_target_in_mem_not_in_top_str))
            # print "\n"	
	    f14.write('%s\n' % batch_active_set[i].strip())

            f11.write('%s\n' % ', '.join(['(%s, %s, %s)' % (unidecode.unidecode(wikidata_id_name_map[tpl_id[0]]), unidecode.unidecode(wikidata_rel_id_name_map[tpl_id[1]]), unidecode.unidecode(wikidata_id_name_map[tpl_id[2]])) for tpl_id in prob_mem_tuples[i] if tpl_id[0] in wikidata_id_name_map and tpl_id[1] in wikidata_rel_id_name_map and tpl_id[2] in wikidata_id_name_map]))
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()
        f7.close()
        f8.close()
	f9.close()
	f10.close()
	f11.close()
	f12.close()
	f13.close()
	f14.close()
	f15.close()

    def map_id_to_word(word_indices, vocab):
        sentence_list = []
        for sent in word_indices:
            word_list = []
            for word_index in sent:
                word = vocab[word_index]
                word_list.append(word)
            sentence_list.append(" ".join(word_list))
        return sentence_list

    def get_predicted_sentence(valid_op, true_op, vocab):
        max_probs_index = []
        max_probs = []
        if true_op is not None:
                true_op = true_op.tolist()
                true_op = np.asarray(true_op).T.tolist()
                true_op_prob = []
        i=0
        for op in valid_op:
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

    test_data = pkl.load(open(param['test_data_file']))
    print 'Test dialogue dataset loaded'
    sys.stdout.flush()
    vocab = pkl.load(open(param['vocab_file'],"rb"))
    vocab_size = len(vocab)
    wikidata_dir = param['wikidata_dir']
    transe_dir = param['transe_dir']
    glove_dir = param['glove_dir']
    wikidata_id_name_map=json.load(open(wikidata_dir+'/items_wikidata_n.json'))
    wikidata_rel_id_name_map=json.load(open('/filtered_property_wikidata4.json'))
    test_text_targets = load_valid_test_target(test_data)
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

    with tf.Graph().as_default():
        model = Hierarchical_seq_model(param['text_embedding_size'], param['wikidata_embed_size'], param['cell_size'], param['cell_type'], param['batch_size'], param['learning_rate'], param['max_len'], param['max_utter'], param['patience'], param['max_gradient_norm'], param['activation'], param['output_activation'],vocab_init_embed, ent_embed, rel_embed, param['memory_size'], param['gold_target_size'], vocab_size)
        model.create_placeholder()
        losses, _, _, logits, prob_mem = model.inference()
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
            batch_target_word_ids = test_text_targets[i*param['batch_size']:(i+1)*param['batch_size']]
            batch_target_sentences = map_id_to_word(batch_target_word_ids, vocab)
            sum_batch_loss = perform_test(model, batch_dict, batch_target_word_ids, batch_target_sentences, vocab, ent_embed, id_entity_map,i, wikidata_id_name_map, wikidata_rel_id_name_map)
            test_loss = test_loss + sum_batch_loss

        print 'Avg. test loss = %f\n' % (float(test_loss)/float(len(test_data)))
        sys.stdout.flush()

        print 'Testing over'


def main():
    param = get_params(sys.argv[1], sys.argv[2], sys.argv[3])
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
