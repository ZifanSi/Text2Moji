import gensim.downloader as api
from gensim.models.fasttext import load_facebook_vectors
import numpy as np
from src.dataset import load_dataset
from nltk.tokenize import TweetTokenizer
import preprocessor as p

tokenizer = TweetTokenizer()

def load_pretrained_embeddings(embedding_name):
    print(f"Loading {embedding_name}...")
    embeddings = api.load(embedding_name)
    print(f"Loaded, Len: {len(embeddings)}, dim: {embeddings.vector_size}")
    return embeddings

def create_embedding_matrix_glove(vocab, pretrained_embeddings):
    embedding_dim = pretrained_embeddings.vector_size
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for word, idx in vocab.items():
        if word in ['<PAD>', '<UNK>']:
            embedding_matrix[idx] = np.zeros((embedding_dim,))
        elif word in pretrained_embeddings:
            embedding_matrix[idx] = pretrained_embeddings[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    pretrained_words = sum(1 for word in vocab if word in pretrained_embeddings)
    coverage = pretrained_words / len(vocab) * 100
    print(f"pretrained embeddings coverage rate: {coverage:.2f}% ({pretrained_words}/{len(vocab)})")
    np.save(f"embedding_matrix_glove_twitter_{embedding_dim}.npy", embedding_matrix)

    return embedding_matrix

def load_fasttext():
    print("Loading fastText...")
    ft = load_facebook_vectors("data/embeddings/crawl-300d-2M-subword.bin")
    return ft

def create_embedding_matrix_fasttext(vocab, pretrained_embeddings):
    embedding_dim = pretrained_embeddings.vector_size
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    for word, idx in vocab.items():
        if word in ['<PAD>', '<UNK>']:
            embedding_matrix[idx] = np.zeros((embedding_dim,))
        else:
            embedding_matrix[idx] = pretrained_embeddings[word]

    np.save("embedding_matrix_crawl_subword_300.npy", embedding_matrix)
    return embedding_matrix

def create_vocab():
    train, test, mapping, output_format = load_dataset()

    token_set = set()

    for text in train['TEXT']:
        text = p.tokenize(text)
        tokens = tokenizer.tokenize(text.lower())
        token_set.update(tokens)

    vocab_list = sorted(list(token_set))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(vocab_list, start=2):
        vocab[word] = i

    print(f"Vocabulary size: {len(vocab)}")

    np.save("vocab.npy", vocab, allow_pickle=True)

    return vocab

if __name__ == "__main__":
    vocab = create_vocab()

    # glove_embeddings = load_pretrained_embeddings("glove-twitter-100")
    # create_embedding_matrix_glove(vocab, glove_embeddings)

    glove_embeddings_200d = load_pretrained_embeddings("glove-twitter-200")
    create_embedding_matrix_glove(vocab, glove_embeddings_200d)

    # fasttext_embeddings = load_fasttext()
    # create_embedding_matrix_fasttext(vocab, fasttext_embeddings)