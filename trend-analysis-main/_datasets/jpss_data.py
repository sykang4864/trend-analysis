import pathlib
import pickle
import re

import pandas as pd
from nltk import pos_tag
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords


####################
# Journal of Payments Strategy & Systems 논문
# - Format : id(eid) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

def _preprocess_text(txt):
    here = pathlib.Path(__file__).resolve().parent
    loc_stopwords = here / 'stopwordsEng.txt'

    txt = txt.lower()
    txt = re.sub('[^A-Za-z0-9가-힣_ ]+', '', txt)

    # tokenize
    words = word_tokenize(txt)

    # stopwords filtering
    stopwords_def = stopwords.words('english')  # nltk default stopwords
    stopwords_ctm = [word.strip() for word in open(loc_stopwords, encoding='utf-8')]
    stopwords_def.extend(stopwords_ctm)
    words_flt = [word for word in words if word not in stopwords_def]

    # select NN*
    words_tag = pos_tag(words_flt)
    words_fin = []
    for word_tag in words_tag:
        if word_tag[1].startswith('NN'):  # select NN, NNS, ...
            words_fin.append(word_tag[0])

    return words_fin


####################
# 데이터 로드 및 전처리
# - 전처리 : Tokenizing, POS tagging & filtering, n-gram, Stopword Filtering
####################
def load_for_keyword(timestamp_name, target_name, timestamp_format='%Y', timestamp_group_format='Y',
                     reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'jpss.csv'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'jpss_pp_for_keyword.pkl', 'rb') as fin:
            doc_group_by_time = pickle.load(fin)
        fin.close()
        return doc_group_by_time

    # 데이터 로드
    df_jpss = pd.read_csv(loc_data)
    df_jpss[timestamp_name] = pd.to_datetime(df_jpss[timestamp_name], format=timestamp_format)
    df_jpss.set_index(timestamp_name, inplace=True)

    # timestamp별 그룹화
    doc_group_by_time = {time: group for time, group in df_jpss.groupby(pd.Grouper(freq=timestamp_group_format))}

    for time in doc_group_by_time.keys():
        target = doc_group_by_time[time][target_name].astype(str).values.tolist()

        # to lowercase
        target = [item.lower() for item in target]

        # 전처리
        documents = []
        for document in target:
            words = _preprocess_text(document)
            if len(words) > 0:
                documents.append(' '.join(words))
        doc_group_by_time[time] = documents

    # 전처리된 결과 저장
    with open(here / 'jpss_pp_for_keyword.pkl', 'wb') as fout:
        pickle.dump(doc_group_by_time, fout)
    fout.close()

    return doc_group_by_time


def load_for_coword(target_index, reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'jpss_pp_for_coword.pkl', 'rb') as fin:
            documents = pickle.load(fin)
        fin.close()
        return documents

    # 데이터 로드
    df_jpss = pd.read_csv(here / 'jpss.csv')
    target = df_jpss.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    documents = []
    for document in target:
        sentences = sent_tokenize(document)
        for sentence in sentences:
            words = _preprocess_text(sentence)
            if len(words) > 0:
                documents.append(' '.join(words))

    # 전처리된 결과 저장
    with open(here / 'jpss_pp_for_coword.pkl', 'wb') as fout:
        pickle.dump(documents, fout)
    fout.close()

    return documents


####################
# Term Weighting을 위한 데이터 로드 및 처리
# - 전처리 : Tokenizing, POS tagging & filtering, Stopword Filtering
####################
def load_for_term_weighting(label_index, target_index):
    here = pathlib.Path(__file__).resolve().parent

    df_jpss = pd.read_csv(here / 'jpss.csv')
    label = df_jpss.iloc[:, label_index].astype(str).values.tolist()
    target = df_jpss.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    documents = []
    for document in target:
        words = _preprocess_text(document)
        if len(words) > 0:
            documents.append(' '.join(words))

    return label, documents


def load_for_topic(timestamp_index, target_index, reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'jpss_pp_for_topic.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents

    # 데이터 로드
    df_jpss = pd.read_csv(here / 'jpss.csv')
    timestamps = df_jpss.iloc[:, timestamp_index].astype(str).values.tolist()
    target = df_jpss.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    documents = []
    for document in target:
        words = _preprocess_text(document)
        if len(words) > 0:
            documents.append(words)

    # 전처리된 결과 저장
    with open(here / 'jpss_pp_for_topic.pkl', 'wb') as fout:
        pickle.dump(timestamps, fout)
        pickle.dump(documents, fout)
    fout.close()

    return timestamps, documents


def load_for_bertopic(timestamp_index, target_index, reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'jpss_pp_for_bert.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents

    # 데이터 로드
    df_jpss = pd.read_csv(here / 'jpss.csv')
    timestamps = df_jpss.iloc[:, timestamp_index].tolist()
    target = df_jpss.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    documents = []
    for document in target:
        words = _preprocess_text(document)
        if len(words) > 0:
            documents.append(' '.join(words))

    # 전처리된 결과 저장
    with open(here / 'jpss_pp_for_bert.pkl', 'wb') as fout:
        pickle.dump(timestamps, fout)
        pickle.dump(documents, fout)
    fout.close()

    return timestamps, documents
