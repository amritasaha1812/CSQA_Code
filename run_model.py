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
import params
import numpy as np
from params import *
import nltk
from read_data import *
from hierarchy_model import *
import gensim
import unidecode

wikidata_id_name_map=json.load(open('/dccstor/cssblr/vardaan/dialog-qa/item_data_filt.json'))
def feeding_dict(model, mem_size, inputs_w2v, inputs_kb, orig_target, target, decoder_target, text_weights, mem_weights, decoder_inputs, sources, rel, key_target, feed_prev, ent_embedding, rel_embedding):
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

    for model_target_i, target_i in zip(model.target_text, target):
        feed_dict[model_target_i] = target_i

    for model_decoder_output_i, target_i in zip(model.decoder_text_outputs, decoder_target):
	feed_dict[model_decoder_output_i] = target_i

    for model_text_weight_i, weight_i in zip(model.text_weights, text_weights):
	feed_dict[model_text_weight_i] = weight_i
    feed_dict[model.mem_weights] = mem_weights

    for model_decoder_input_i, decoder_input_i in zip(model.decoder_text_inputs, decoder_inputs):
	feed_dict[model_decoder_input_i] = decoder_input_i

    #for model_source, source in zip(model.sources, sources):
    #    feed_dict[model_source] = source

    #for model_source_emb, source in zip(model.sources_emb, sources):
    #    feed_dict[model_source_emb] = np.array([ent_embedding[i] for i in source], dtype=np.float32)
    feed_dict[model.sources_emb] = np.array([np.array([ent_embedding[i] for i in source]) for source in sources])

    #for model_rel, relation in zip(model.rel, rel):
    #    feed_dict[model_rel] = relation
	
    #for model_rel_emb, rel in zip(model.rel_emb, rel):
    #	feed_dict[model_rel_emb] = np.array([rel_embedding[i] for i in rel], dtype=np.float32)
    feed_dict[model.rel_emb] = np.array([np.array([rel_embedding[i] for i in rel_i]) for rel_i in rel])
 
    #for model_key_target, key_target_i in zip(model.key_target, key_target):
    #    feed_dict[model_key_target] = key_target_i

    #for model_target_emb, key_target_i in zip(model.key_target_emb, key_target):
    #    feed_dict[model_target_emb] = np.array([ent_embedding[i] for i in key_target_i], dtype=np.float32)
    feed_dict[model.key_target_emb] = np.array([np.array([ent_embedding[i] for i in key_target_i]) for key_target_i in key_target])
 
    for model_gold_emb, orig_target_i in zip(model.gold_emb, orig_target):
	feed_dict[model_gold_emb] = np.array([ent_embedding[i] for i in orig_target_i], dtype=np.float32)
 	
    feed_dict[model.feed_previous] = feed_prev
    return feed_dict

def check_dir(param):
    if not os.path.exists(param['logs_path']):
        os.makedirs(param['logs_path'])
    if not os.path.exists(param['model_path']):
        os.makedirs(param['model_path'])

def get_test_op(sess, model, batch_dict, step, ent_embedding):
    memory_size, test_batch_enc_w2v, test_batch_enc_kb, batch_target, batch_orig_target, batch_response, batch_orig_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, batch_active_set = get_batch_data(param['max_len'], param['max_utter'], param['memory_size'], param['gold_target_size'], param['batch_size'], batch_dict)
    feed_dict = feeding_dict(model, memory_size, test_batch_enc_w2v, test_batch_enc_kb, batch_orig_target, batch_target, batch_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, True, ent_embedding, rel_embedding)
    dec_op, loss, loss_decoder, loss_kvmem= sess.run([logits, losses, losses_decoder, losses_kvmem], feed_dict=feed_dict)
    return loss, loss_decoder, loss_kvmem, dec_op

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

def perform_test(sess, model, saver, model_file, get_pred_sentence, param, logits, losses, vocab):
    print 'reading model from  modelfile'
    saver.restore(sess, model_file)
    test_data = pkl.load(open(param['test_data_file']),'rb')
    print "Test dialogues loaded"
    predicted_sentence = []
    test_loss = 0
    n_batches = len(test_data)/param['batch_size']
    test_text_targets = read_data_task1.load_valid_test_target(param['test_data_file'])
    for i in range(n_batches):
        batch_dict = test_data[i*param['batch_size']:(i+1)*param['batch_size']]
        test_op, sum_batch_loss = get_test_op(sess, model, batch_dict, param, logits, losses)
        test_loss = sum_batch_loss + test_loss
        predicted_sentence.append(get_predicted_sentence(test_op, None, vocab)[0])
    test_predicted_sentence = predicted_sentence[0:len(test_text_targets)]
    write_to_file(test_predicted_sentence, test_text_targets)
    print ('average test loss is =%.6f' %(float(test_loss)/float(len(test_data))))
    sys.stdout.flush()

def run_training(param):

    def get_train_loss(model, batch_dict, feed_prev, step, ent_embedding, rel_embedding, type_of_loss):
	if type_of_loss == "decoder":
		overriding_memory = 10
	else:
		overriding_memory = None
        memory_size, train_batch_enc_w2v, train_batch_enc_kb, batch_target, batch_orig_target, batch_response, batch_orig_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, batch_active_set = get_batch_data(param['max_len'], param['max_utter'], param['memory_size'], param['gold_target_size'], param['batch_size'], batch_dict, overriding_memory)
        feed_dict = feeding_dict(model, memory_size, train_batch_enc_w2v, train_batch_enc_kb, batch_orig_target,  batch_target, batch_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, feed_prev, ent_embedding, rel_embedding)
	if type_of_loss == "decoder":
	        output_loss, output_prob, _ = sess.run([losses, prob, train_op], feed_dict=feed_dict)
	elif type_of_loss == "kvmem":
		output_loss, output_prob, _ = sess.run([losses, prob, train_op], feed_dict=feed_dict)
        return output_loss, output_prob

    def get_valid_loss(model, batch_dict, ent_embedding, rel_embedding, id_entity_map, type_of_loss):
	if type_of_loss == "decoder":
		overriding_memory = 10
	else:
		overriding_memory = None
        memory_size, val_batch_enc_w2v, val_batch_enc_kb, batch_target, batch_orig_target, batch_response, batch_orig_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, batch_active_set = get_batch_data(param['max_len'], param['max_utter'], param['memory_size'], param['gold_target_size'], param['batch_size'], batch_dict, overriding_memory)
        feed_dict = feeding_dict(model, memory_size, val_batch_enc_w2v, val_batch_enc_kb, batch_orig_target, batch_target, batch_response, batch_text_weight, batch_mem_weight, batch_decoder_input, batch_sources, batch_rel, batch_key_target, True, ent_embedding, rel_embedding)
	if type_of_loss == "decoder":
		output_loss, output_prob = sess.run([losses, prob], feed_dict)
		return output_loss, output_prob, batch_orig_response
	elif type_of_loss == "kvmem":
		output_loss, output_prob = sess.run([losses, prob], feed_dict)
		prob_mem_entries = output_prob	
        	prob_mem_entries = np.array(prob_mem_entries)
	 	mem_entries_sorted = np.fliplr(np.argsort(prob_mem_entries,axis=1))
		mem_attention_sorted = np.fliplr(np.sort(prob_mem_entries,axis=1))
		#batch_inverse_mapped_target is of dimension batch_size * mem_size + 1, each entry being the global index
		mem_entries_sorted = [[batch_key_target[j][i] for j in mem_entries_sorted[i]] for i in range(batch_key_target.shape[1])]
		mem_entries_sorted = np.array(mem_entries_sorted, dtype=np.int32)
		mem_entries_sorted = [[id_entity_map[mem_entries_sorted[i][j]] for j in range(mem_entries_sorted.shape[1])] for i in range(mem_entries_sorted.shape[0])]
		gold_entity_ids = np.transpose(batch_orig_target, (1,0))
		gold_entity_ids = [[id_entity_map[gold_entity_ids[i][j]] for j in range(gold_entity_ids.shape[1])] for i in range(gold_entity_ids.shape[0])]
	        return output_loss, mem_entries_sorted, mem_attention_sorted, gold_entity_ids, batch_orig_response

    def get_sum_batch_loss(batch_loss):
	batch_loss = np.asarray(batch_loss)
	batch_loss[np.where(batch_loss>100.)]=0.
        return np.sum(batch_loss)

    def perform_training(model, batch_dict, feed_prev, step, ent_embedding, rel_embedding, type_of_loss):
	if type_of_loss == "decoder":
		batch_train_loss, dec_op = get_train_loss(model, batch_dict, feed_prev, step, ent_embedding, rel_embedding, type_of_loss)
	elif type_of_loss == "kvmem":
		batch_train_loss, output_prob = get_train_loss(model, batch_dict, feed_prev, step, ent_embedding, rel_embedding, type_of_loss)
        sum_batch_loss = get_sum_batch_loss(batch_train_loss)
        return sum_batch_loss

    def perform_evaluation(model, batch_dict, batch_target_word_ids, batch_text_targets, epoch, step, vocab, ent_embedding, rel_embedding, id_entity_map, type_of_loss):
	if type_of_loss == "decoder":
		batch_valid_loss, valid_op, batch_orig_response = get_valid_loss(model, batch_dict, ent_embedding, rel_embedding, id_entity_map, type_of_loss)
		batch_predicted_sentence, prob_predicted_words, prob_true_words = get_predicted_sentence(valid_op, batch_target_word_ids, vocab)	
		print_pred_true_op_decoder(batch_predicted_sentence, prob_predicted_words, prob_true_words, batch_text_targets, step, epoch, batch_valid_loss)
	elif type_of_loss == "kvmem":
		batch_valid_loss, prob_mem_entries, prob_mem_scores, gold_entity_ids, gold_orig_response = get_valid_loss(model, batch_dict, ent_embedding, rel_embedding, id_entity_map, type_of_loss)
		print_pred_true_op_kvmem(prob_mem_entries, prob_mem_scores, gold_entity_ids, gold_orig_response, step, epoch, batch_valid_loss)
        sum_batch_loss = get_sum_batch_loss(batch_valid_loss)
        return sum_batch_loss

    def evaluate(model, epoch, step, valid_data, valid_text_targets, vocab, ent_embedding, rel_embedding, id_entity_map, loss_type):
        print 'Validation started'
        sys.stdout.flush()
        valid_loss = 0.
        batch_predicted_sentence=[]
        n_batches = int(math.ceil(float(len(valid_data))/float(param['batch_size'])))
        for i in range(n_batches):
            batch_dict = valid_data[i*param['batch_size']:(i+1)*param['batch_size']]
            batch_target_word_ids = valid_text_targets[i*param['batch_size']:(i+1)*param['batch_size']]
            batch_target_sentences = map_id_to_word(batch_target_word_ids, vocab)
            sum_batch_loss = perform_evaluation(model, batch_dict, batch_target_word_ids, batch_target_sentences, epoch, step, vocab, ent_embedding, rel_embedding, id_entity_map, loss_type)
            valid_loss = valid_loss + sum_batch_loss
        return float(valid_loss)/float(len(valid_data))

    def print_pred_true_op_decoder(pred_op, prob_pred, prob_true, true_op, step, epoch, batch_valid_loss):
        for i in random.sample(xrange(len(true_op)),5):
	    if i==len(true_op):	
		continue
            print "true sentence in step "+str(step)+" of epoch "+str(epoch)+" is:"
            sys.stdout.flush()
	    print true_op[i]
            print "predicted sentence in step "+str(step)+" of epoch "+str(epoch)+" is:"
	    sys.stdout.flush()
            print pred_op[i]
	    print "prob of predicted words in step "+str(step)+" of epoch "+str(epoch)+" is:"
            sys.stdout.flush()
	    print prob_pred[i]
            print "prob of true words in step "+str(step)+" of epoch "+str(epoch)+" is:"
	    sys.stdout.flush()
            print prob_true[i]
	    print "loss for the pair", str(batch_valid_loss[i])

    def print_pred_true_op_kvmem(prob_memory_entities, prob_memory_scores, gold_entity_ids, gold_orig_response, step, epoch, batch_valid_loss):	
	for i in random.sample(xrange(len(gold_orig_response)),5):
	    if i==len(gold_orig_response):
		continue
	    print "top-5 memory entries in step "+str(step)+" of epoch "+str(epoch)+" is:"
            x = [unidecode.unidecode(mem_entry) for mem_entry in prob_memory_entities[i] if mem_entry not in ['<pad_kb>','<nkb>','<unk>']]
		
	    print [wikidata_id_name_map[xi] for xi in x if xi in wikidata_id_name_map][:5]
	    print "top-5 KB entities in step "+str(step)+" of epoch "+str(epoch)+" is:"
	    print [wikidata_id_name_map[id] for id in prob_memory_entities[i] if id in wikidata_id_name_map if wikidata_id_name_map[id] not in ['<pad_kb>','<nkb>','<unk>']][:5]
	    print "probability distribution over the top-10 memory entries "
            print prob_memory_scores[i][:10]
            print "gold entities in step "+str(step)+" of epoch "+str(epoch)+" is:"
	    print [wikidata_id_name_map[id] for id in gold_entity_ids[i] if id in wikidata_id_name_map]
	    print "gold response in step "+str(step)+" of epoch "+str(epoch)+" is:"
            print gold_orig_response[i]
            sys.stdout.flush()
            print "loss for the pair", str(batch_valid_loss[i])
            print "\n"

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
        #len(valid_op) is max_len
        #true_op is of dimension batch_size * max_len
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

    train_data = []
    if not isinstance(param['train_data_file'], list):
	training_files = [param['train_data_file']]
    else:
	training_files = param['train_data_file']
	random.shuffle(training_files)	
    #for file in training_files:
    #	train_data.extend(pkl.load(open(file)))
    print 'Train dialogue dataset loaded'
    sys.stdout.flush()
    valid_data =[]
    if not isinstance(param['valid_data_file'], list):
	valid_files = [param['valid_data_file']]
    else:
	valid_files = param['valid_data_file']
    for file in valid_files:
	print file
	valid_data.extend(pkl.load(open(file)))
    print 'Valid dialogue dataset loaded'
    sys.stdout.flush()
    vocab = pkl.load(open(param['vocab_file'],"rb"))
    response_vocab = pkl.load(open(param['response_vocab_file'],"rb"))
    vocab_size = len(vocab)
    response_vocab_size = len(response_vocab)
    valid_text_targets = load_valid_test_target(valid_data)
    print 'valid target sentence list loaded'
    print 'writing terminal output to file'
    f_out = open(param['terminal_op'],'w')
    sys.stdout=f_out
    check_dir(param)
    model_file = os.path.join(param['model_path'],"best_model")

    vocab_init_embed = np.empty([len(vocab.keys()), param['text_embedding_size']],dtype=np.float32)
    #word2vec_pretrain_embed = gensim.models.KeyedVectors.load_word2vec_format('/dccstor/anirlaha1/data/GoogleNews-vectors-negative300.bin', binary=True)
    word2vec_pretrain_embed = gensim.models.Word2Vec.load_word2vec_format('/dccstor/anirlaha1/data/GoogleNews-vectors-negative300.bin', binary=True)
    # word2vec_pretrain_embed = {} # to be removed later

    ent_embed = np.load('/dccstor/cssblr/vardaan/projE-wikidata/ProjE/ent_embed.pkl.npy')
    rel_embed = np.load('/dccstor/cssblr/vardaan/projE-wikidata/ProjE/rel_embed.pkl.npy')

    new_row = np.zeros((1,param['wikidata_embed_size']), dtype=np.float32)
    
    ent_embed = np.vstack([new_row, ent_embed]) # corr. to <pad_kb>
    ent_embed = np.vstack([new_row, ent_embed]) # corr. to <nkb>

    rel_embed = np.vstack([new_row, rel_embed]) # corr. to <pad_kb>
    rel_embed = np.vstack([new_row, rel_embed]) # corr. to <nkb>
    #ent_embed = numpy.vstack([ent_embed, newrow]) # extra entry in kb-vocab corr. to non-kb words

    for i in xrange(vocab_init_embed.shape[0]):
        if vocab[i] in word2vec_pretrain_embed:
            vocab_init_embed[i,:] = word2vec_pretrain_embed[vocab[i]]
        elif i == 4: # KB word
	    vocab_init_embed[i,:] = np.zeros((1,vocab_init_embed.shape[1]),dtype=np.float32)
            #vocab_init_embed[i,:] = np.zeros(1,vocab_init_embed.shape[1])
        else:
	    vocab_init_embed[i,:] = np.random.rand(1,vocab_init_embed.shape[1]).astype(np.float32)
            #vocab_init_embed[i,:] = np.random.rand(1,vocab_init_embed.shape[1])
   
    id_entity_map = {0:'<pad_kb>', 1: '<nkb>'}
    id_entity_map.update({(k+2):v for k,v in pkl.load(open('/dccstor/cssblr/vardaan/projE-wikidata/ProjE/id_entity_map.pickle','rb')).iteritems()}) 
    type_of_loss = ""
    if 'type_of_loss' in param:
	type_of_loss=param['type_of_loss']
    print 'type of loss ', type_of_loss
    with tf.Graph().as_default():
        model = Hierarchical_seq_model(param['text_embedding_size'], param['wikidata_embed_size'], param['cell_size'], param['cell_type'], param['batch_size'], param['learning_rate'], param['max_len'], param['max_utter'], param['patience'], param['max_gradient_norm'], param['activation'], param['output_activation'],vocab_init_embed, ent_embed, rel_embed, param['gold_target_size'], response_vocab_size, type_of_loss)
        model.create_placeholder()
        losses, prob = model.inference()
        # losses = model.loss_task_text(logits)
        train_op, gradients = model.train(losses)
        print "model created"
        sys.stdout.flush()
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        if len(os.listdir(param['model_path']))>0:
            print "best model exists.. restoring from that point"
            saver.restore(sess, model_file)
        else:
            print "initializing fresh variables"
            sess.run(init)
        best_valid_loss = float("inf")
        best_valid_epoch=0
        all_var = tf.all_variables()
        print 'printing all' , len(all_var),' TF variables:'
        for var in all_var:
            print var.name, var.get_shape()
        print 'training started'
        sys.stdout.flush()
        last_overall_avg_train_loss = None
        overall_step_count = 0
        for epoch in range(param['max_epochs']):
	  len_train_data = 0.
          train_loss=0.
	  for file in training_files:
	    train_data = pkl.load(open(file))
	    len_train_data = len_train_data + len(train_data)
	    random.shuffle(train_data)
	    train_data.sort(key=lambda x : len(x[6].split("|"))) #filter(lambda a: a!= kb_pad_idx, x[6].split("|"))))
	    print 'sorted training data by memory size'	
	    n_batches = int(math.ceil(float(len(train_data))/float(param['batch_size'])))
    	    print 'number of batches ', n_batches, 'len train data ', len(train_data), 'batch size' , param['batch_size']
	    sys.stdout.flush()
            for i in range(n_batches):
		if 'feed_prev' in param:
			feed_prev = param['feed_prev']
		elif epoch>0 or overall_step_count>5000:
                       feed_prev = True
                else:
                       feed_prev = False
                train_batch_dict=train_data[i*param['batch_size']:(i+1)*param['batch_size']]
                sum_batch_loss = perform_training(model, train_batch_dict, feed_prev, overall_step_count, ent_embed, rel_embed, type_of_loss)
                avg_batch_loss = sum_batch_loss / float(param['batch_size'])
                if overall_step_count%param['print_train_freq']==0:
                    print('Epoch  %d Step %d train loss (avg over batch) =%.6f' %(epoch, i, avg_batch_loss))
                    sys.stdout.flush()
                train_loss = train_loss + sum_batch_loss
                avg_train_loss = float(train_loss)/float(i+1)
                if overall_step_count > 0 and overall_step_count%param['valid_freq']==0:
                    overall_avg_valid_loss = evaluate(model, epoch, i, valid_data, valid_text_targets, response_vocab, ent_embed, rel_embed, id_entity_map, type_of_loss)
                    print ('Epoch %d Step %d ... overall avg valid loss= %.6f ' %(epoch, i, overall_avg_valid_loss))
                    sys.stdout.flush()
                    if best_valid_loss>overall_avg_valid_loss:
                        saver.save(sess, model_file)
                        best_valid_loss=overall_avg_valid_loss
		overall_step_count = overall_step_count + 1
          overall_avg_train_loss = train_loss/float(len_train_data)
          print 'epoch ',epoch,' of training is completed ... overall avg. train loss ', overall_avg_train_loss
          if last_overall_avg_train_loss is not None and overall_avg_train_loss > last_overall_avg_train_loss:
              diff = overall_avg_train_loss - last_overall_avg_train_loss
              if diff>param['train_loss_incremenet_tolerance']:
                      print 'WARNING: training loss (%.6f) has increased by %.6f since last epoch, has exceed tolerance of %f ' %(overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance'])
              else:
                      print 'WARNING: training loss (%.6f) has increased by %.6f since last epoch, but still within tolerance of %f ' %(overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance'])
          last_overall_avg_train_loss = overall_avg_train_loss
          sys.stdout.flush()
        print 'Training over'
        print 'Evaluating on test data'
    f_out.close()



def main():
    try:
        ques_type_id = int(sys.argv[2])
    except:
        ques_type_id = -1
    param = get_params(sys.argv[1])	
    if 'feed_prev' in param and not param['feed_prev']:
	print "'feed_prev' in param and not param['feed_prev']"	
	#sys.exit(1)	
    # param = get_params(os.getcwd())
    if not os.path.exists(os.path.join(os.getcwd(), sys.argv[1], 'dump')):
        os.makedirs(os.path.join(os.getcwd(), sys.argv[1], 'dump'))

    if not os.path.exists(os.path.join(os.getcwd(), sys.argv[1], 'model')):
        os.makedirs(os.path.join(os.getcwd(), sys.argv[1], 'model'))

    if not os.path.exists(os.path.join(os.getcwd(), sys.argv[1], 'log')):
        os.makedirs(os.path.join(os.getcwd(), sys.argv[1], 'log'))

    print 'dump dir created'
    print param
    if isinstance(param['train_data_file'], list) and isinstance(param['valid_data_file'], list) and all([os.path.exists(x) for x in param['train_data_file']]) and all([os.path.exists(x) for x in param['valid_data_file']]):
	print 'dictionary already exists'
        sys.stdout.flush()		
    elif isinstance(param['train_data_file'], str) and isinstance(param['valid_data_file'], str) and os.path.exists(param['train_data_file']) and os.path.exists(param['valid_data_file']):# and os.path.exists(param['test_data_file']):
        print 'dictionary already exists'
        sys.stdout.flush()
    else:
        get_dialog_dict(param)
        print 'dictionary formed'
        sys.stdout.flush()
    run_training(param)

if __name__=="__main__":
    main()
