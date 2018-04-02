import gensim
import cPickle as pkl
word2vec_pretrain_embed = gensim.models.Word2Vec.load_word2vec_format('/dccstor/cssblr/amrita/resources/glove/GoogleNews-vectors-negative300.bin', binary=True)
from annoy import AnnoyIndex

f=300
index = AnnoyIndex(f, metric='euclidean')
index_desc = {}
vocab = pkl.load(open('vocab_count.pkl'))
count = 0
for word in vocab:
        word = word[0]
        if word in word2vec_pretrain_embed:
                if word in word2vec_pretrain_embed:
                        embed = word2vec_pretrain_embed[word]
                        index.add_item(count, embed)
                        index_desc[count] = word
                        count = count+1
index.build(100)
index.save('annoy_index/glove_embedding_of_vocab.ann')
pkl.dump(index_desc, open('annoy_index/index2word.pkl','wb'))
