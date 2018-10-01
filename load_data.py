import numpy as np
from module import *
import prep

prepath_data = './aclImdb/movie_data/'
tp_filename = 'full_train_pos.txt'   #train_pos_filename
tn_filename = 'full_train_neg.txt'   #train_neg_filename
tep_filename = 'full_test_pos.txt'    #test_pos_filename
ten_filename = 'full_test_neg.txt'    #test_neg_filename



movie_train_data, movie_train_target = read_data(prepath_data + tp_filename, prepath_data + tn_filename)
movie_test_data, movie_test_target = read_data(prepath_data + tep_filename, prepath_data + ten_filename)

train_split = 20000
train_data = movie_train_data[:train_split]
train_targets = movie_train_target[:train_split]

val_data = movie_train_data[train_split:]
val_targets = movie_train_target[train_split:]

test_data = movie_test_data
test_targets = movie_test_target

print ("Train dataset size is {}".format(len(train_data)))
print ("Val dataset size is {}".format(len(val_data)))
print ("Test dataset size is {}".format(len(test_data)))



import spacy
import string

# Load English tokenizer, tagger, parser, NER and word vectors
tokenizer = spacy.load('en_core_web_sm')
punctuations = string.punctuation

# lowercase and remove punctuation
def tokenize(sent):
    tokens = tokenizer(sent)
    return [token.text.lower() for token in tokens if (token.text not in punctuations)]

# Example
tokens = tokenize(u'Apple is looking at buying U.K. startup for $1 billion')
print (tokens)

# This is the code cell that tokenizes train/val/test datasets
# However it takes about 15-20 minutes to run it
# For convinience we have provided the preprocessed datasets
# Please see the next code cell
import pickle as pkl

def tokenize_dataset(dataset):
    token_dataset = []
    # we are keeping track of all tokens in dataset 
    # in order to create vocabulary later
    all_tokens = []
    
    for sample in dataset:
        tokens = tokenize(sample)
        token_dataset.append(tokens)
        all_tokens += tokens

    return token_dataset, all_tokens

#val set tokens
print ("Tokenizing val data")
val_data_tokens, _ = tokenize_dataset(val_data)
pkl.dump(val_data_tokens, open(prepath_data + "val_data_tokens.p", "wb"))

#test set tokens
print ("Tokenizing test data")
test_data_tokens, _ = tokenize_dataset(test_data)
pkl.dump(test_data_tokens, open(prepath_data + "test_data_tokens.p", "wb"))

#train set tokens
print ("Tokenizing train data")
train_data_tokens, all_train_tokens = tokenize_dataset(train_data)
pkl.dump(train_data_tokens, open(prepath_data + "train_data_tokens.p", "wb"))
pkl.dump(all_train_tokens, open(prepath_data + "all_train_tokens.p", "wb"))