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



