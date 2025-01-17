import os
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from stemming.porter2 import stem
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import normalize
from itertools import groupby
import pickle

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import PySimpleGUI as sg


class DocumentSplitter:
    def split_dump_to_documents(self, file_path, dir_path):
        with open(file_path, 'r') as file:
            text = file.readlines()
        articles = [list(group) for key, group in groupby(text, lambda x: x == "\n") if not key]
        for article in articles:
            path = dir_path + article[0].strip().translate(str.maketrans('\\/', '__')) + '.txt'
            with open(path, 'w+') as doc:
                for line in article:
                    doc.write(line)


class SearchEngine:
    def __init__(self):
        # key - name of article : val - dictionary of dicts (word : frequency) for each document
        # de facto sparse matrix A, but stored as a dictionary of dictionaries
        self.document_word_frequency_dicts = {}
        # key - word : number of occurrences in all documents
        self.word_all_occurrences_count = {}
        # key - word : number of occurrences in all documents -reduced
        self.word_all_occurrences_count_reduced = {}
        # key - word : number of documents in which occurred
        self.word_documents_count = {}

        self.word_row_index = {}
        self.document_column_index = {}
        self.document_column_index_to_name = {}

        self.loaded_documents_count = 0
        self.start_time = datetime.now()
        self.k_most_common_words = 20000
        self.A_matrix = None
        self.A_matrix_SVD = None
        self.svd = None

    def load_documents_from_dir(self, dir_path):
        for filename in os.listdir(dir_path):
            with open(dir_path + '/' + filename) as file:
                self.__parse_document_to_frequency_dict(file)

    def __parse_document_to_frequency_dict(self, file):
        # store content of document - without stop words, punctuation
        #                               as dictionary
        file_content = file.read().translate(str.maketrans('', '', string.punctuation))
        self.loaded_documents_count += 1
        if self.loaded_documents_count % 1000 == 0:
            print('scanned', self.loaded_documents_count, 'took: ', datetime.now() - self.start_time)

        word_tokens = map(lambda x: x.lower(), word_tokenize(file_content))
        word_tokens_filtered = [w for w in word_tokens if w not in set(stopwords.words('english'))]

        word_tokens_stemmed = [stem(w) for w in word_tokens_filtered]

        document_word_freq = dict()

        for word in word_tokens_stemmed:
            document_word_freq[word] = document_word_freq.get(word, 0) + 1

        self.document_word_frequency_dicts[str(file.name)] = document_word_freq
        self.__add_to_bag_of_words(document_word_freq)
        self.__inc_word_documents_count(document_word_freq)

    def __add_to_bag_of_words(self, document_word_freq):
        for word in document_word_freq:
            self.word_all_occurrences_count[word] = self.word_all_occurrences_count.get(word, 0) \
                                                    + document_word_freq[word]

    def __inc_word_documents_count(self, document_word_freq):
        for word in document_word_freq:
            self.word_documents_count[word] = self.word_documents_count.get(word, 0) + 1

    def save_all_dicts(self, dir_path):
        names = SearchEngine().__dict__.keys()
        # print(names)
        for name in names:
            with open(dir_path + '/' + name + '.pkl', 'wb') as f:
                pickle.dump(getattr(self, name), f, pickle.HIGHEST_PROTOCOL)

    def load_all_dicts(self, dir_path):
        names = SearchEngine().__dict__.keys()
        # load only existing attributes
        names_in_dir = [w for w in names if w + '.pkl' in os.listdir(dir_path)]
        print(names_in_dir)
        for name in names_in_dir:
            with open(dir_path + '/' + name + '.pkl', 'rb') as f:
                setattr(self, name, pickle.load(f))

    # take k most frequent words from union of words in all documents
    def reduce_bag_of_words(self):
        self.word_all_occurrences_count_reduced = dict(Counter(self.word_all_occurrences_count) \
                                                       .most_common(self.k_most_common_words))

    def set_row_index_to_words(self):
        self.word_row_index = {}
        for i, w in enumerate(self.word_all_occurrences_count_reduced):
            self.word_row_index[w] = i

    def set_column_index_to_documents(self):
        self.document_column_index = {}
        self.document_column_index_to_name = {}
        for i, d in enumerate(self.document_word_frequency_dicts):
            self.document_column_index[d] = i
            self.document_column_index_to_name[i] = d

    def init_matrix(self):
        self.A_matrix = lil_matrix((len(self.word_all_occurrences_count_reduced), self.loaded_documents_count),
                                   dtype=np.double)
        print('Initializing matrix of shape:', self.A_matrix.shape)
        i = 0
        for w in self.word_row_index:
            for d in self.document_column_index:
                if w in self.document_word_frequency_dicts[d]:
                    self.A_matrix[self.word_row_index[w], self.document_column_index[d]] \
                        = np.double(self.document_word_frequency_dicts[d].get(w, 0)) * self.__get_idf(w)
            i += 1
            if i % 1000 == 0:
                print('Parsed words', i)
        # to make scalar product faster
        self.A_matrix = csr_matrix(self.A_matrix)
        self.A_matrix = self.__normalize_matrix(self.A_matrix)

    def __get_idf(self, word):
        return np.log(self.loaded_documents_count / self.word_documents_count[word])

    def __normalize_matrix(self, a):
        # axis=0 - by column, axis=1 - by row
        return normalize(a, norm='l2', axis=0)

    def parse_query_to_vec(self, query):
        query = query.translate(str.maketrans('', '', string.punctuation))
        word_tokens = map(lambda x: x.lower(), word_tokenize(query))
        word_tokens_filtered = [w for w in word_tokens if w not in set(stopwords.words('english'))]
        word_tokens_stemmed = [stem(w) for w in word_tokens_filtered]
        word_tokens_in_set = [w for w in word_tokens_stemmed if w in self.word_row_index]

        Q = lil_matrix((len(self.word_all_occurrences_count_reduced), 1), dtype=np.double)

        for word in word_tokens_in_set:
            Q[self.word_row_index[word], 0] = 1

        Q = csr_matrix(Q)
        Q = self.__normalize_matrix(Q)
        return Q

    def get_correlation_of_query(self, query):
        Q = self.parse_query_to_vec(query)
        return Q.T.dot(self.A_matrix)

    def get_best_matched_documents(self, k, query):
        corr = self.get_correlation_of_query(query).toarray()
        id_corr = [(i, corr[0, i]) for i in range(self.loaded_documents_count)]
        id_corr = sorted(id_corr, key=(lambda x: x[1]), reverse=True)
        id_corr = id_corr[:k]
        print('\nregular search')
        layout_results = []
        for i, el in enumerate(id_corr):
            article_name = self.document_column_index_to_name[el[0]].replace('../data/', '').replace('.txt', '')
            correlation = el[1] * 100
            hex_red = str(hex(int(correlation * 2.55))).replace('0x', '').upper()
            hex_red = '0' + hex_red if len(hex_red) == 1 else hex_red
            layout_results.append([sg.Button(article_name, button_color=(
                'black', '#' + hex_red + '7F7F')),
                                   sg.Text(str(correlation))])
        return layout_results

    def get_matrix_svd(self):
        self.svd = TruncatedSVD(n_components=250)
        self.svd.fit(self.A_matrix)
        self.A_matrix_SVD = self.svd.transform(self.A_matrix)

    def get_correlation_of_query_svd(self, query):
        Q_SVD = self.parse_query_to_vec(query).T.dot(self.A_matrix_SVD)
        res = Q_SVD.dot(self.svd.components_)
        return res

    def get_best_matched_documents_svd(self, k, query):
        corr = self.get_correlation_of_query_svd(query)
        id_corr = [(i, corr[0, i]) for i in range(self.loaded_documents_count)]
        id_corr = sorted(id_corr, key=(lambda x: x[1]), reverse=True)
        id_corr = id_corr[:k]
        print('\nSVD search')
        layout_results = []
        for i, el in enumerate(id_corr):
            article_name = self.document_column_index_to_name[el[0]].replace('../data/', '').replace('.txt', '')
            correlation = el[1] * 100
            hex_red = str(hex(int(correlation * 2.55))).replace('0x', '').upper()
            hex_red = '0' + hex_red if len(hex_red) == 1 else hex_red
            layout_results.append([sg.Button(article_name, button_color=(
                'black', '#' + hex_red + '7F7F')),
                                   sg.Text(str(correlation))])
        return layout_results


if __name__ == '__main__':

    doc_splitter = DocumentSplitter()
    # doc_splitter.split_dump_to_documents('example.txt', './example_dir/')
    # doc_splitter.split_dump_to_documents('corpus.txt', './data/')

    search_engine = SearchEngine()
    # search_engine.load_documents_from_dir('../data_test')
    # search_engine.load_documents_from_dir('../data')

    # search_engine.reduce_bag_of_words()
    # search_engine.word_all_occurrences_count_reduced = search_engine.word_all_occurrences_count

    # search_engine.set_row_index_to_words()
    # search_engine.set_column_index_to_documents()
    #
    # search_engine.init_matrix()
    # search_engine.get_matrix_svd()

    # search_engine.load_all_dicts('obj_reduced_20k')
    search_engine.load_all_dicts('obj')

    while True:
        svd_button = sg.Checkbox('With SVD', default=False)
        layout = [[sg.Image(filename='loupe.png')],  # Icon made by Freepik from www.flaticon.com
                  [sg.Text('Enter query below')],
                  [sg.Input(do_not_clear=True)],
                  [svd_button],
                  [sg.Button('Search',
                             button_color=('black', '#FFFFFF')),
                   sg.Exit()]]

        query_window = sg.Window('SVD Search Engine').Layout(layout)
        event, query = query_window.Read()
        query = query[0]
        svd_button_state = svd_button.Get()
        query_window.Close()
        if event is 'Exit' or event is None:
            break
        else:
            if svd_button_state:
                layout_results = search_engine.get_best_matched_documents_svd(10, query)
            else:
                layout_results = search_engine.get_best_matched_documents(10, query)
            results_window = sg.Window('Results').Layout(layout_results)
            event_res = results_window.Read()
            if event_res[0] is not None:
                with open('../data/' + event_res[0] + '.txt', 'r') as file:
                    sg.Popup(event_res[0], file.read(), line_width=200)
