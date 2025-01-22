from submission import count_cooccur_matrix, cooccur_to_embedding, top_k_similar
from util import *

def test_embedding(words=['man', 'woman', 'happy', 'sad', 'emma', 'knightley']):
    tokens = read_corpus()
    word2ind, co_mat = count_cooccur_matrix(tokens, window_size=1)
    embeddings = cooccur_to_embedding(co_mat, 100)
    for word in words:
        word_ind = word2ind[word]
        top_k_words = top_k_similar(word_ind, embeddings, word2ind, k=10, metric='cosine')
        print('top k most similar words to', word)
        print(' '.join(top_k_words))

test_embedding()


