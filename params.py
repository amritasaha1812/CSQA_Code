import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
import os
def get_params(dir):
    param={}
    dir= str(dir)
    param['train_dir_loc']="/dccstor/cssblr/vardaan/dialog-qa/CSQA_v7/train"
    param['valid_dir_loc']="/dccstor/cssblr/vardaan/dialog-qa/CSQA_v7/valid/"
    param['test_dir_loc']="/dccstor/cssblr/vardaan/dialog-qa/CSQA_v7/test/"
    param['wikidata_dir']="/dccstor/cssblr/vardaan/dialog-qa/"
    param['transe_dir']="transe_dir/"
    param['lucene_dir']="lucene_dir/"
    param['glove_dir']="/dccstor/cssblr/amrita/resources/glove/"
    param['dump_dir_loc']=dir+"/dump/"
    param['test_output_dir']=dir+"/test_output/"
    param['vocab_file']=dir+"/vocab.pkl"
    param['train_data_file']=[dir+"/dump/"+x for x in os.listdir(dir+"/dump") if x.startswith('train_')]
    param['valid_data_file']=[dir+"/dump/"+x for x in os.listdir(dir+"/dump") if x.startswith('valid_')]
    param['test_data_file']=dir+"/dump/test_data_file.pkl"
    param['vocab_file']=dir+"/vocab.pkl"
    param['response_vocab_file']=dir+"/response_vocab.pkl"
    param['vocab_stats_file']=dir+"/vocab_stats.pkl"
    param['model_path']=dir+"/model"
    param['terminal_op']=dir+"/terminal_output.txt"
    param['logs_path']=dir+"/log"
    param['type_of_loss']="decoder"
    param['text_embedding_size'] = 300
    param['activation'] = None #tf.tanh
    param['output_activation'] = None #tf.nn.softmax
    param['cell_size']= 512
    param['cell_type']=rnn_cell.GRUCell
    param['batch_size']=64
    param['vocab_freq_cutoff']=5
    param['learning_rate']=0.0004
    param['patience']=200
    param['early_stop']=100
    param['max_epochs']=1000000
    param['max_len']=20
    param['max_utter']=2
    param['print_train_freq']=100
    param['show_grad_freq']=20
    param['valid_freq']=5000
    param['max_gradient_norm']=0.1
    param['train_loss_incremenet_tolerance']=0.01
    param['wikidata_embed_size']= 100
    param['memory_size'] = 50000
    param['gold_target_size'] = 10
    param['stopwords'] = 'stopwords.pkl'
    param['stopwords_histogram'] = 'stopwords_histogram.txt'
    param['vocab_max_len'] = 40000
    return param
