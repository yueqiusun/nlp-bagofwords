import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize   

  
def remove_stopwords(l):
    stop_words = set(stopwords.words('english')) 
    l_s = l.split(' ')
    filtered_sentence = [w for w in l_s if not w in stop_words] 
    return filtered_sentence

def prep_data(r):
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    r1 = [REPLACE_NO_SPACE.sub("", line.lower()) for line in r]
    r2 = [REPLACE_WITH_SPACE.sub(" ", line) for line in r1]
    r3 = [remove_stopwords(line) for line in r2]
    output = r3
    return output
    
def word_grams(sent, min=1, max=4):
    words = sent.split(' ')
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


def read_data(filename_p, filename_n, count = 9999999):
    x = []
    y = []
    raw_data_p = []
    
    with open(filename_p, "r") as f:
        line_num_p = 0
        line_p = f.readline()
        while line_p != None and line_p != "" and line_num_p<count:
            print(line_p)
            line_num_p += 1
            raw_data_p.append(line_p)
            line_p = f.readline()
    raw_data_n = []
    with open(filename_n, "r") as f:
        line_num_n = 0
        line_n = f.readline()
        while line_n != None and line_n != "" and line_num_n<count:
            line_num_n += 1
            raw_data_n.append(line_n)
            line_n = f.readline()
            
    x = x + prep_data(raw_data_p)
    x = x + prep_data(raw_data_n)
    y = y + [1] * line_num_p
    y = y + [0] * line_num_n
    return x, y