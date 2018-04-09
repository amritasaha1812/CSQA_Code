import numpy as np
import math
import sys
import os
from cStringIO import StringIO
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
from tensorflow.python.ops import control_flow_ops
import seq2seq
from seq2seq import *
from tensorflow.python.ops import variable_scope
sys.path.append(os.getcwd())
from seq2seq import *
class Hierarchical_seq_model():
    def __init__(self, text_embedding_size, wikidata_embed_size, cell_size, cell_type, batch_size, learning_rate, max_len, max_utter, patience, max_gradient_norm, activation, output_activation, vocab_init_embed, ent_embed, rel_embed, max_target_size, decoder_words, type_of_loss=""):
        self.text_embedding_size = text_embedding_size
        self.wikidata_embed_size = wikidata_embed_size
        self.cell_size = cell_size
        self.cell_type = cell_type
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate),trainable=False)
        self.max_len = max_len
        self.max_utter = max_utter
        self.patience = patience
	self.decoder_words = decoder_words
        self.max_gradient_norm = max_gradient_norm
    
        self.encoder_text_inputs = None
        self.decoder_text_inputs = None
	self.decoder_text_outputs = None
        self.target_text = None
        self.text_weights = None
	self.mem_weights = None	
        self.feed_previous = None
        #self.sources = None #new
        #self.rel = None #new
	#self.rel_emb = None #new
        #self.key_target = None #new
        self.gold_emb = None
        self.activation = activation
        self.output_activation = output_activation

        self.enc_scope_text = None #scope for the sentence level encoder for text
        self.enc_cells_utter = None #encoder cells (utterenace context level), at this level it is multimodal
        self.enc_scope_utter = None #scope for the utterance level encoder
	self.dec_cells_text=None#List for different decoder cells(different languages) for now only one.
        self.dec_scope_text=None#scope for the sentence decoder
        self.vocab_init_embed = vocab_init_embed
        self.hops = 2
        self.max_target_size = max_target_size
        self.dropout_memory = 0.3
	self.type_of_loss = type_of_loss
	self.eps = 1e-7

        initializer = tf.contrib.layers.xavier_initializer()
        self.R_1 = tf.Variable(initializer([self.wikidata_embed_size, self.cell_size]), name = 'H')
        self.B = tf.Variable(initializer([self.cell_size, self.wikidata_embed_size]), name = 'B')
        self.C = tf.Variable(initializer([self.cell_size, 2 * self.wikidata_embed_size]), name = 'C')
        # loading of wikidata embeddings (to be done later)

        def create_cell_scopes():
            self.embeddings = tf.get_variable('embedding_matrix',initializer=self.vocab_init_embed,dtype=tf.float32)
	    self.rel_embeddings = tf.get_variable('rel_embedding_matrix', initializer=rel_embed, dtype=tf.float32, trainable=False)
            self.enc_scope_text = "encoder_text"
            self.enc_cells_utter = self.cell_type(self.cell_size)
            self.enc_scope_utter = "encoder_utter"
    	    self.dec_cells_text = self.cell_type(self.cell_size)
            self.dec_scope_text = "decoder_text"
	
        create_cell_scopes()

    def create_placeholder(self):
	self.memory_size = tf.placeholder(tf.int32, shape=[], name="memory_size")	
        self.encoder_text_inputs_w2v = [[tf.placeholder(tf.int32,[None], name="encoder_text_inputs_w2v") for i in range(self.max_len)] for j in range(self.max_utter)] # list of list of tensor placeholders; altogether of dimension batch_size * max_utter * max_len
        self.encoder_text_inputs_kb = [[tf.placeholder(tf.int32,[None], name="encoder_text_inputs_kb") for i in range(self.max_len)] for j in range(self.max_utter)]
	self.encoder_text_inputs_kb_emb = [[tf.placeholder(tf.float32, [None,self.wikidata_embed_size], name="encoder_text_inputs_kb_emb") for i in range(self.max_len)] for j in range(self.max_utter)] 
	self.decoder_text_inputs = [tf.placeholder(tf.int32,[None], name="decoder_text_inputs") for i in range(self.max_len)]
	self.text_weights = [tf.placeholder(tf.float32, [None], name="text_weights") for i in range(self.max_len)]
	self.mem_weights = tf.placeholder(tf.float32,[self.batch_size, None], name="mem_weights")
        self.decoder_text_outputs = [tf.placeholder(tf.int32,[None], name="decoder_text_outputs") for i in range(self.max_len)]
        self.feed_previous = tf.placeholder(tf.bool, name='feed_previous')
        self.target_text = [tf.placeholder(tf.int32,[None], name="target_text") for i in range(self.max_target_size)]
	self.sources_emb = tf.placeholder(tf.float32, [None, self.batch_size, self.wikidata_embed_size], name='sources_emb')
	self.rel_emb = tf.placeholder(tf.float32, [None, self.batch_size, self.wikidata_embed_size], name='rel_emb')
	self.key_target_emb = tf.placeholder(tf.float32, [None, self.batch_size, self.wikidata_embed_size], name='key_target_emb')
    	self.gold_emb = [tf.placeholder(tf.float32, [None, self.wikidata_embed_size], name='gold_emb') for i in range(self.max_target_size)]

    def hierarchical_encoder(self):
        n_steps = self.max_len
        enc_text_states = self.sentence_encoder(self.encoder_text_inputs_w2v, self.encoder_text_inputs_kb_emb)
        enc_utter_states = self.utterance_encoder(enc_text_states)
        return enc_utter_states

    def sentence_encoder(self, enc_inputs_w2v, enc_inputs_kb_emb):
        # for the sentence level encoder: enc_inputs is of dimension (max_utter, max_len, batch_size)
        utterance_states = []
        with tf.variable_scope(self.enc_scope_text) as scope:
            #init_state = self.enc_cells_text.zero_state(self.batch_size, tf.float32)
            for i in range(0, len(enc_inputs_w2v)):
                if i>0:
                    scope.reuse_variables()
                #enc_inputs[i] is a max_len sized list of tensor of dimension (batch_size)
                rnn_inputs_w2v = tf.nn.embedding_lookup(self.embeddings, tf.pack(self.encoder_text_inputs_w2v[i],axis=1))
		rnn_inputs_kb = tf.pack(enc_inputs_kb_emb[i], axis=0)
		rnn_inputs_kb = tf.transpose(rnn_inputs_kb, perm=[1,0,2]) 
                rnn_inputs = tf.concat(2, [rnn_inputs_w2v, rnn_inputs_kb])
		
                cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
                init_state = tf.get_variable('init_state', [1, self.cell_size], initializer=tf.constant_initializer(0.0),dtype=tf.float32)
                init_state = tf.tile(init_state, [self.batch_size, 1])
                _, states = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=init_state)
                #rnn.rnn takes a max_len sized list of tensors of dimension (batch_size * self.text_embedding_size) (after passing through the embedding wrapper)
                #states is of dimension (batch_size, cell_size)
                utterance_states.append(states)
        #utterance_states is of dimension (max_utter, batch_size, cell_size)        
        return utterance_states

    def utterance_encoder(self, enc_inputs):
        # for the utterance level encoder: enc_inputs is of dimension (max_utter, batch_size, cell_size+max_images*image_embedding_size)
        utterance_states =  None
        utterance_outputs = None
        with tf.variable_scope(self.enc_scope_utter) as scope:
            cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
            init_state = tf.get_variable('init_state', [1, self.cell_size], initializer=tf.constant_initializer(0.0))
            init_state = tf.tile(init_state, [self.batch_size, 1])
            outputs, states = tf.nn.dynamic_rnn(cell, tf.pack(enc_inputs,axis=1), initial_state=init_state)
            #rnn.rnn takes a max_utter sized list of tensors of dimension (batch_size * cell_size+(max_images*image_embedding_size))
            utterance_states= states
            utterance_outputs = outputs
        # utterance_states is of dimension (batch_size, cell_size)    
        top_states = [array_ops.reshape(e, [-1, 1, self.enc_cells_utter.output_size]) for e in tf.unpack(utterance_outputs,axis=1)]
        attention_states = array_ops.concat(1, top_states)
        return utterance_states, attention_states

    def tf_print(self, x):
        old_stdout =sys.stdout
        sys.stdout= mystdout = StringIO()
        shape_dec = x.get_shape()
        sys.stdout= old_stdout
    
    def kv_memNN_encoder(self,ques_utterance_state):
        q_0 = ques_utterance_state #batch_size * (cell_size)
        q = [q_0]

        for hop in xrange(self.hops):
	    keys_emb = tf.pack(self.sources_emb, axis=0)
            keys_emb = tf.transpose(keys_emb, perm=[1,0,2])
	    rel_emb = tf.transpose(self.rel_emb, perm=[1,0,2])
            #rel_emb = tf.nn.embedding_lookup(self.rel_embeddings, tf.pack(self.rel, axis=1))  # batch_size * size_memory * wikidata_embed_size
            k = tf.concat(2, [keys_emb, rel_emb]) # batch_size * size_memory * (2*wikidata_embed_size)
           
	    ones = tf.ones([self.memory_size, 1], tf.float32)
            ones_dropout = tf.nn.dropout(ones, self.dropout_memory, noise_shape=[self.memory_size, 1])
 
            # q[-1] shape: batch_size * (text_embedding_size + wikidata_embed_size)
            q_last = tf.matmul(q[-1], self.C) # batch_size * (2*wikidata_embed_size)
            q_temp = tf.expand_dims(q_last,-1) # batch_size * (2*wikidata_embed_size) * 1
            q_temp1 = tf.transpose(q_temp, [0, 2, 1])  # batch_size * 1 * (2*wikidata_embed_size)

            prod = k * q_temp1  # batch_size * size_memory * (2*wikidata_embed_size)
            dotted = tf.reduce_sum(prod, 2) # batch_size * size_memory
	    #probs = tf.nn.softmax(tf.multiply(dotted, self.mem_weights))
	    #probs = tf.nn.softmax(tf.multiply(dotted, self.mem_weights))
            probs = tf.nn.softmax(tf.multiply(dotted, self.mem_weights))
	    '''
            probs = tf.nn.softmax(dotted)
            probs = tf.multiply(probs, self.mem_weights)
	    num = len(probs.get_shape())
            l1norm = tf.reduce_sum(probs, axis=1)
	    stacked_norm = tf.multiply(tf.ones_like(probs), tf.expand_dims(l1norm, axis=num-1))
	    probs = tf.where(tf.equal(stacked_norm, 0.), tf.ones_like(probs), probs)
            new_l1norm = tf.reduce_sum(probs, axis=1)
	    probs = probs/tf.reshape(new_l1norm, (-1,1))
	    '''
	    values_emb = tf.pack(self.key_target_emb, axis=0)
	    values_emb = tf.transpose(values_emb, perm=[1,0,2])
            
	    #apply dropout on values
            values_emb_dropout = values_emb * ones_dropout

            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1]) #batch_size * 1 * size_memory
            v_temp = tf.transpose(values_emb_dropout, [0,2,1]) #batch_size * wikidata_embed_size * size_memory
            o_k = tf.reduce_sum(v_temp * probs_temp, 2) #batch_size * wikidata_embed_size
	
	    o_k = tf.matmul(o_k, self.R_1)
            q_k = q[-1]+ o_k #batch_size * cell_size
            q.append(q_k)
	#q_k is of dimension #batch_size * wikidata_embed_size   
        return q_k

    def decode(self, concatenated_word_input, loop_fn, dec_cell, initial_state, utterance_output, dec_scope):
	    state = initial_state
	    outputs = []
	    prev = None
	    for i in range(len(concatenated_word_input)):
		inp_word = concatenated_word_input[i]
		if loop_fn is not None and prev is not None:
			with tf.variable_scope("loop_function", reuse=True):
				inp_word = loop_fn(prev, i)
				inp_word = tf.concat(1, [utterance_output, inp_word])
		if i > 0:
			dec_scope.reuse_variables()
		output, state = dec_cell(inp_word, state, scope=dec_scope)
		outputs.append(output)
		if loop_fn is not None:
			prev = output
	    return outputs, state		 					

    def decoder(self, decoder_inputs, utterance_outputs):
	with tf.variable_scope(self.dec_scope_text) as scope:
		init_state = self.dec_cells_text.zero_state(self.batch_size, tf.float32)
		max_val = np.sqrt(6. / (self.decoder_words + self.cell_size))
		weight_word = tf.get_variable("dec_weights",[self.cell_size,self.decoder_words],initializer=tf.random_uniform_initializer(-1.*max_val,max_val))
		bias_word = tf.get_variable("dec_biases",[self.decoder_words],initializer=tf.constant_initializer(0.0))
		def feed_previous_decode(feed_previous_bool):
			dec_embed_word, loop_fn = seq2seq.get_decoder_embedding(decoder_inputs, self.decoder_words, self.text_embedding_size, output_projection=(weight_word,bias_word), feed_previous=feed_previous_bool)	
			concatenated_input_word = self.get_dec_concat_ip(dec_embed_word, utterance_outputs)
			dec_output_word, _ = self.decode(concatenated_input_word, loop_fn, self.dec_cells_text, init_state, utterance_outputs, scope)
			return dec_output_word
					
		dec_output_word = control_flow_ops.cond(self.feed_previous, lambda: feed_previous_decode(True), lambda: feed_previous_decode(False))
		for i in range(len(dec_output_word)):
			dec_output_word[i] = tf.matmul(dec_output_word[i], weight_word) + bias_word
			if self.output_activation is not None:	
				dec_output_word[i] = self.output_activation(dec_output_word[i]) #batch_size * decoder_words
	return dec_output_word

    def get_dec_concat_ip(self, dec_embed, utterance_output):
        concat_dec_inputs = []
        for (i, inp) in enumerate(dec_embed):
                #inp is of dimension batch_size * self.text_embedding_size 
                #utterance_output is of dimension  batch_size * cell_size
                concat_dec_inputs.append(tf.concat(1, [utterance_output, inp]))
                #self.concat_dec_inputs[i] is of dimension batch_size * (cell_size + self.text_embedding_size)                                  
        #self.concat_dec_inputs is of dimension max_len * batch_size * (cell_size + self.text_embedding_size)
        return concat_dec_inputs
    
    def loss_kvmem(self, kv_memNN_encoder_op):
        temp_1 = tf.matmul(kv_memNN_encoder_op, self.B) # (q_k * B) # batch_size * wikidata_embed_size

	values_emb = tf.pack(self.key_target_emb, axis=0)
        psi = tf.transpose(values_emb, perm=[1,0,2])
        #psi = tf.nn.embedding_lookup(self.ent_embeddings, tf.pack(self.key_target, axis=1)) # batch_size * size_memory * embedding_size

        temp_1_expand = tf.expand_dims(temp_1, -1) # batch_size * wikidata_embed_size * 1
        temp_1_expand = tf.transpose(temp_1_expand, [0, 2, 1])  # batch_size * 1 * wikidata_embed_size

        temp_2 = temp_1_expand * psi # batch_size * size_memory * embedding_size

        prob_mem = tf.reduce_sum(temp_2, 2) # batch_size * size_memory

        temp_3 = tf.one_hot(tf.pack(self.target_text,axis=1), depth=self.memory_size, axis=1) # batch_size * size_memory * max_target_size
        temp_4 = tf.reduce_sum(temp_3, axis=2)
	temp_4 = tf.multiply(temp_4, self.mem_weights)
        loss = tf.nn.softmax_cross_entropy_with_logits(prob_mem, temp_4) 
        return loss, prob_mem
    

    def inference(self):
	utterance_output, attention_states = self.hierarchical_encoder()
	kv_memNN_encoder_op = self.kv_memNN_encoder(utterance_output)
	if self.type_of_loss=="decoder":
		concat_kvmem_utterance = tf.concat(1, [utterance_output, kv_memNN_encoder_op])
	        logits = self.decoder(self.decoder_text_inputs, concat_kvmem_utterance)
		losses_decoder = self.loss_decoder(logits)
		losses = losses_decoder
		prob = tf.nn.softmax(logits)
		return losses, prob
	elif self.type_of_loss=="kvmem":
		losses_kvmem, prob_mem = self.loss_kvmem(kv_memNN_encoder_op)
		losses = losses_kvmem
		prob = tf.nn.softmax(prob_mem)
		return losses, prob
    '''
    def inference(self):
        utterance_output, attention_states = self.hierarchical_encoder()
        # KV-MemNN code inserted here
        kv_memNN_encoder_op = self.kv_memNN_encoder(utterance_output)
	concat_kvmem_utterance = tf.concat(1, [utterance_output, kv_memNN_encoder_op])
	logits = self.decoder(self.decoder_text_inputs, concat_kvmem_utterance) 
        losses_kvmem, prob_mem = self.loss_kvmem(kv_memNN_encoder_op)
	losses_decoder = self.loss_decoder(logits)
	if self.type_of_loss=="decoder":
		losses = losses_decoder
	elif self.type_of_loss=="kvmem":
		losses = losses_kvmem
	else:
		losses = losses_decoder + losses_kvmem
	prob_mem = tf.nn.softmax(prob_mem)
	logits = tf.nn.softmax(logits)
        return losses,losses_decoder,losses_kvmem, logits, prob_mem
    '''

    def loss_decoder(self, logits):
        #logits is a max_len sized list of 2-D tensors of dimension batch_size * decoder_words
        #self.target_text is a max_len sized list of 1-D tensors of dimension batch_size
        #self.text_weights is a max_len sized list of 1-D tensors of dimension batch_size
        losses=seq2seq.sequence_loss_by_example(logits, self.decoder_text_outputs, self.text_weights)
        #losses is a 1-D tensor of dimension batch_size 
        return losses

    def train(self, losses):
        parameters=tf.trainable_variables()
        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
        gradients=tf.gradients(losses,parameters)
        #print tf.get_default_graph().as_graph_def()
        clipped_gradients,norm=tf.clip_by_global_norm(gradients,self.max_gradient_norm)
        global_step=tf.Variable(0,name="global_step",trainable='False')
        #train_op=optimizer.minimize(losses,global_step=global_step)
        train_op=optimizer.apply_gradients(zip(clipped_gradients,parameters),global_step=global_step)
        return train_op, clipped_gradients
