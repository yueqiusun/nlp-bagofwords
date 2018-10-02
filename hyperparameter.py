class Hyperparameter:
	# save index 0 for unk and 1 for pad
	PAD_IDX = 0
	UNK_IDX = 1
	MAX_SENTENCE_LENGTH = 200
	prepath_data = './aclImdb/movie_data/'
	tp_filename = 'full_train_pos.txt'   #train_pos_filename
	tn_filename = 'full_train_neg.txt'   #train_neg_filename
	tep_filename = 'full_test_pos.txt'    #test_pos_filename
	ten_filename = 'full_test_neg.txt'    #test_neg_filename