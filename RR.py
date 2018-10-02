import numpy as np
from module import Module
import pickle as pkl
import prep
from hyperparameter import Hyperparameter as hp

def load_raw_data():



	movie_train_data, movie_train_target = prep.read_data(hp.prepath_data + hp.tp_filename, hp.prepath_data + hp.tn_filename)
	movie_test_data, movie_test_target = prep.read_data(hp.prepath_data + hp.tep_filename, hp.prepath_data + hp.ten_filename)

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

	return train_data, train_targets, val_data, val_targets, test_data, test_targets


def tokenization(train_data,val_data,test_data):
	#val set tokens
	print ("Tokenizing val data")
	val_data_tokens, _ = prep.tokenize_dataset(val_data)
	pkl.dump(val_data_tokens, open(hp.prepath_data + "val_data_tokens.p", "wb"))

	#test set tokens
	print ("Tokenizing test data")
	test_data_tokens, _ = prep.tokenize_dataset(test_data)
	pkl.dump(test_data_tokens, open(hp.prepath_data + "test_data_tokens.p", "wb"))

	#train set tokens
	print ("Tokenizing train data")
	train_data_tokens, all_train_tokens = prep.tokenize_dataset(train_data)
	pkl.dump(train_data_tokens, open(hp.prepath_data + "train_data_tokens.p", "wb"))
	pkl.dump(all_train_tokens, open(hp.prepath_data + "all_train_tokens.p", "wb"))


def load_tokens():

	# load preprocessed train, val and test datasets
	train_data_tokens = pkl.load(open(hp.prepath_data + "train_data_tokens.p", "rb"))
	all_train_tokens = pkl.load(open(hp.prepath_data + "all_train_tokens.p", "rb"))

	val_data_tokens = pkl.load(open(hp.prepath_data + "val_data_tokens.p", "rb"))
	test_data_tokens = pkl.load(open(hp.prepath_data + "test_data_tokens.p", "rb"))

	# double checking
	print ("Train dataset size is {}".format(len(train_data_tokens)))
	print ("Val dataset size is {}".format(len(val_data_tokens)))
	print ("Test dataset size is {}".format(len(test_data_tokens)))

	print ("Total number of tokens in train dataset is {}".format(len(all_train_tokens)))

	return train_data_tokens, val_data_tokens, test_data_tokens, all_train_tokens

#token2id, id2token = Module.build_vocab(all_train_tokens)