from collections import Counter

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time
import numpy as np
from keras.utils import np_utils
import pickle
import sys
import os
import re
#import warnings
#warnings.filterswarnings('ignore')

class Data_loader():

    def __init__(self):

        self.max_length = 3000
        self.word_to_index_dict = dict()
        self.output_length = 0
        self.keyword_path = 'D:/keyword_labels/'
        self.summary_path = 'D:/summary_labels/'
        self.characters = re.compile(r'[ㄱ-ㅣ가-힣a-zA-Z0-9 ]+')
        self.bracket = re.compile('\(.*?\)')
        self.keyword_labels = os.listdir(self.keyword_path)[:10000]
        self.summary_labels = os.listdir(self.summary_path)[:10000]

###################################################################################################


    def is_characters(self, sent):

        return self.characters.search(sent) is not None

###################################################################################################
    # 입력 데이터를 가지고 문서 초록의 단어 리스트의 리스트와 초록 단어 모든 리스트, 초록 등을 반환한다.
    def make_dataset(self):
        start_time = time.time()
        summary_list = []
        keyword_list = []
        keyList = []
        for summary_file in self.summary_labels:
            summary_ok = False
            with open(self.summary_path + summary_file, 'r', encoding="UTF-8") as f:
                summaries = f.read()
                summaries = self.bracket.sub("", summaries)
                if summaries in summary_list: continue
                if summaries:
                    summary_ok = True
                    summary_list.append(summaries.strip())

            if not summary_ok:
                continue

            with open(self.keyword_path + summary_file, 'r', encoding="UTF-8") as f:
                keysentence = f.read()
                keywords = [self.bracket.sub("", keyword) for keyword in keysentence.split('\n') if self.is_characters(keyword)]
                keywords = [keyword for keyword in keywords if keyword]
                #keyword_list = keyword_list + keywords
                keyList.append(keywords)

        All_keyword_list = [word for keywords in keyList for word in keywords]
        All_keyword_list = list(set(All_keyword_list))                          # 레이블의 단어 중복을 제거한다.
        All_keyword_list = [word for word in All_keyword_list if word]

        output_summary = './data/summary_list.pkl'
        with open(output_summary, 'wb') as f:
            pickle.dump(summary_list, f)
        output_keyword = './data/keyword_list.pkl'
        with open(output_keyword, 'wb') as f:
            pickle.dump(keyList, f)
        output_all_keyword = './data/all_key_word_list.pkl'
        with open(output_all_keyword, 'wb') as f:
            pickle.dump(All_keyword_list, f)
        self.output_length = len(All_keyword_list)
        sys.stderr.write("Make " + output_summary + " (" + str(time.time() - start_time) + "`s), " + output_keyword +" (" + str(time.time() - start_time) + "`s), " + output_all_keyword +" (" + str(time.time() - start_time) + "`s),\n")
        print(len(summary_list), "," , len(keyList), ",", len(All_keyword_list))
        return summary_list, keyList, All_keyword_list

###################################################################################################
    #  keyList, label_list으로 멀티핫 코딩을 한다.
    def make_multihot_list(self, keyList, label_list, output_file="./data/multihot_list.pkl"):
        start_time = time.time()
        #multihot_list = []
        multihot_list = np.zeros((len(keyList), len(label_list)), dtype='int32')

        for i in range(len(keyList)):
            #multilabel = []
            for keyword in keyList[i]:
                #multilabel.append(label_list.index(keyword))
                multihot_list[i][label_list.index(keyword)] = 1
            #multihot_list.append(multilabel)

        with open(output_file, 'wb') as f:
            pickle.dump(multihot_list, f)
        sys.stderr.write("Make " + output_file + " (" + str(time.time() - start_time) + "`s)\n")

        return multihot_list

###################################################################################################
    # make_eumjeol 은 문장을 한 글자 한 글자 잘라서 숫자로 변환한다.
    def make_eumjeol(self, summary_list, output_file="./data/eumjeol_dict.pkl"):
        start_time = time.time()

        sent_list = [sent.strip() for sent in summary_list]
        sent_len = [len(sent) for sent in sent_list]
        # max_length = np.array(sent_len).max()
        sent_list = [self.pad_sentence([c for c in sent]) for sent in sent_list]
        num = 0
        eumjeol_dict = {}
        for word in sent_list:
            for eumjeol in word:
                if eumjeol in eumjeol_dict: continue
                eumjeol_dict[eumjeol] = num
                num = num + 1

        with open(output_file, 'wb') as f:
            pickle.dump(eumjeol_dict, f)
        output_sent = './data/sent_list.pkl'
        with open(output_sent, 'wb') as f:
            pickle.dump(sent_list, f)

        self.word_to_index_dict = eumjeol_dict

        sys.stderr.write("Make " + output_sent + " (" + str(time.time() - start_time) + "`s)" + output_file + " (" + str(time.time() - start_time) + "`s)\n")

        return sent_list, eumjeol_dict
#################################################################################################
    # sentence리스트(한 글자 한 글자)를 받아서 padding 한다.
    def pad_sentence(self, sentence, padding_word="<PAD/>"):
        num_padding = self.max_length - len(sentence)
        if num_padding < 0:
            sentence = sentence[:num_padding]
            num_padding = 0
        sentence_padded = sentence + [padding_word] * num_padding

        return sentence_padded

#################################################################################################
    # train_data 를 만든다.
    def make_train_data(self, sent_list, eumjeol_dict, output_file='./data/train_data.pkl'):

        start_time = time.time()

        train_data = []
        for sent in sent_list:
            sentence = []
            for c in sent:
                sentence = sentence + [eumjeol_dict[c]]
            train_data.append(sentence)

        with open(output_file, 'wb') as f:
            pickle.dump(train_data, f)

        sys.stderr.write("Make " + output_file + " (" + str(time.time() - start_time) + "`s)\n")

        return train_data

##################################################################################################

    def data_load(self):
        start_time = time.time()

        summary_list, keyList, All_keyword_list = self.make_dataset()
        multihot_list = self.make_multihot_list(keyList, All_keyword_list)
        sent_list, eumjeol_dict = self.make_eumjeol(summary_list)
        train_data = self.make_train_data(sent_list, eumjeol_dict)
        train_data = np.array(train_data)
        multihot_list = np.array(multihot_list)
        sys.stderr.write("Make train DATA" + " (" + str(time.time() - start_time) + "`s)\n")
        return train_data, multihot_list


    def data_load2(self, train_file, label_file):
        output_all_keyword = './data/all_key_word_list.pkl'
        with open(output_all_keyword, 'rb') as f:
            all_keyword_list = pickle.load(f)
            self.output_length = len(all_keyword_list)
        output_dict = "./data/eumjeol_dict.pkl"
        with open(output_dict, 'rb') as f:
            eumjeol_dict = pickle.load(f)
            self.word_to_index_dict = eumjeol_dict
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)
            train_data = np.array(train_data)
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
            label_data = np.array(label_data)

        return train_data, label_data
##################################################################################################

if __name__ == '__main__':

    dl = Data_loader()
    x_train, y_train =  dl.data_load()


