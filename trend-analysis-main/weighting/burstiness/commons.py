import platform
from copy import deepcopy

from datetime import datetime, timedelta
from dateutil import parser

import pandas as pd

import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt

import time
from collections import defaultdict

from matplotlib.legend_handler import HandlerLine2D
from sklearn.feature_extraction.text import CountVectorizer

# Machine learning
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

import tomotopy as tm

import pickle

significance_threshold = 0.0002
time_stamp_above_significance = 3
testing_period = 3

# Detection threshold is set such that the top 500 terms are chosen
# burstiness_threshold_detection = 0.020275   # 499
burstiness_threshold_detection = 0.026975   # 100

plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rc('font', family='sans-serif')

def calc_macd(dataset, long_ma_length, short_ma_length, significance_ma_length, signal_line_ma):
    long_ma = dataset.ewm(span=long_ma_length).mean()
    short_ma = dataset.ewm(span=short_ma_length).mean()
    significance_ma = dataset.ewm(span=significance_ma_length).mean()
    macd = short_ma - long_ma
    signal = macd.ewm(span=signal_line_ma).mean()
    hist = macd - signal
    return long_ma, short_ma, significance_ma, macd, signal, hist

def calc_significance(stacked_vectors, significance_threshold, n):
    # Must have been above the significance threshold for two consecutive timesteps
    a = stacked_vectors > significance_threshold
    b = a.rolling(window=n).sum()
    return stacked_vectors[stacked_vectors.axes[1][np.where(b.max() >= n)[0]]]

def calc_burstiness(hist, long_ma_length, scaling_factor):
    return hist.iloc[long_ma_length - 1:] / scaling_factor

def calc_scaling(significance_ma, significance_ma_length, method):
    if method == "max":
        scaling = significance_ma.iloc[significance_ma_length - 1:].max()
    elif method == "mean":
        scaling = significance_ma.iloc[significance_ma_length - 1:].mean()
    elif method == "sqrt":
        scaling = np.sqrt(significance_ma.iloc[significance_ma_length - 1:].max())
    return scaling

def max_burstiness(burstiness, absolute=False):
    if absolute:
        b = pd.concat([np.abs(burstiness).max(), burstiness.idxmax()], axis=1)
    else:
        b = pd.concat([burstiness.max(), burstiness.idxmax()], axis=1)
    b.columns = ["max", "location"]
    return b

def get_prevalence(cluster, bursts, burstvectors, unique_time_stamp):
    indices = []
    for term in cluster:
        if term in bursts:
            indices.append(bursts.index(term))

    prevalence = []
    for time_stamp in unique_time_stamp:
        prevalence.append(
            100 * np.sum(np.sum(burstvectors[time_stamp][:, indices], axis=1) > 0) / burstvectors[time_stamp].shape[0])

    return prevalence

# Term burstiness 계산
# - MACD 곡선 = 단기 이동평균선 - 장기 이동평균선
# - MACD 곡선이 시그널 곡선을 상향 돌파하면 골든 크로스
# - MACD 곡선이 시그널 곡선을 하향 돌파하면 데드 크로스
def compute_term_burstiness(df, date_index, target_index, min_time_stamp_df, long_ma_length, short_ma_length, significance_ma_length, signal_line_ma):
    # Build a vocabulary
    # We have to build a vocabulary before we vectorise the data. This is because we want to set limits on the size of the vocabulary.

    vocab = set()

    grouped_df = df.groupby(df.columns[date_index])

    print('##### vocab generation start')
    for key, item in grouped_df:
        a_group = grouped_df.get_group(key)

        # The same as above, applied time stamp by time stamp instead.
        t0 = time.time()

        vectorizer = CountVectorizer(min_df=min_time_stamp_df)
        vector = vectorizer.fit_transform(a_group.iloc[:, target_index])

        # Save the new words
        vocab = vocab.union(vectorizer.vocabulary_.keys())
        time_stamp = a_group.iloc[:, date_index].tolist()[0]
        # print('time stamp: {}\tvocab: {}\telapsed time: {}'.format(time_stamp, len(vocab), time.time() - t0))

    vocabulary = {}
    i = 0
    for v in vocab:
        vocabulary[v] = i
        i += 1

    print('##### total vocab size: {}'.format(len(vocabulary.keys())))

    # Go time stamp by time stamp and vectorise based on our vocabulary
    # We read in the cleaned data and vectorise it according to our vocabulary.
    vectors = []
    grouped_df = df.groupby(df.columns[date_index])

    print('##### document-term matrix generation start')
    for key, item in grouped_df:
        a_group = grouped_df.get_group(key)

        # The same as above, applied time stamp by time stamp instead.
        t0 = time.time()

        vectorizer = CountVectorizer(vocabulary=vocabulary)
        vector = vectorizer.fit_transform(a_group.iloc[:, target_index])
        vectors.append(vector)
        time_stamp = a_group.iloc[:, date_index].tolist()[0]
        # print('time stamp: {}\telapsed time: {}'.format(time_stamp, time.time() - t0))

    # Summing the vectors
    # We sum the vectors along columns, so that we have the popularity of each term in each time stamp.
    print('##### document-term matrix normalize start')
    summed_vectors = []
    for y in range(len(vectors)):
        vector = vectors[y]
        # Set all elements that are greater than one to one -- we do not care if a word is used multiple times in
        # the same document
        vector[vector > 1] = 1

        # Sum the vector along columns
        summed = np.squeeze(np.asarray(np.sum(vector, axis=0)))

        # Normalise by dividing by the number of documents in that time stamp
        normalised = summed / vector.shape[0]

        # Save the summed vector
        summed_vectors.append(normalised)

    # Stack vectors vertically, so that we have the full history of popularity/time for each term
    stacked_vectors = np.stack(summed_vectors, axis=1)

    print('##### term-timestamp matrix shape: {}'.format(stacked_vectors.shape))

    stacked_vectors = pd.DataFrame(stacked_vectors.transpose(), columns=list(vocabulary.keys()))

    normalisation = stacked_vectors.sum(axis=1)
    stacked_vectors = stacked_vectors.divide(normalisation, axis='index') * 100

    stacked_vectors = calc_significance(stacked_vectors, significance_threshold, time_stamp_above_significance)
    print('##### significance calc result matrix shape: {}'.format(stacked_vectors.shape))

    with open('./models/burstiness_stacked_vec.pickle', 'wb') as handle:
        pickle.dump(stacked_vectors, handle)

    # Calculate burstiness
    print('##### macd calc start')
    long_ma, short_ma, significance_ma, macd, signal, hist \
        = calc_macd(stacked_vectors, long_ma_length, short_ma_length, significance_ma_length, signal_line_ma)
    scaling_factor = calc_scaling(significance_ma, significance_ma_length, "sqrt")
    print('##### burstiness calc start')
    burstiness_over_time = calc_burstiness(hist, long_ma_length, scaling_factor)
    burstiness = max_burstiness(burstiness_over_time)
    # max value 기준 정렬
    burstiness.sort_values(by=['max'], axis=0, ascending=False, inplace=True)

    # print('##### burstiness result (term, max value, max time stamp)')
    # print(burstiness)

    return burstiness, burstiness_over_time

def topic_modeling(burstiness, df, date_index, target_index):
    # Set a threshold such that the top 500 bursty terms are included
    print(np.sum(burstiness["max"] > 0.026975))

    bursts = list(burstiness["max"].index[np.where(burstiness["max"] > burstiness_threshold_detection)[0]])

    # Cluster bursts based on co-occurence
    # vectorise again, using these terms only
    burstvectors = {}
    grouped_df = df.groupby(df.columns[date_index])

    corpus = tm.utils.Corpus()

    unique_time_stamp = []
    for key, item in grouped_df:
        a_group = grouped_df.get_group(key)
        # The same as above, applied time stamp by time stamp instead.
        t0 = time.time()

        vectorizer = CountVectorizer(vocabulary=bursts)
        vector = vectorizer.fit_transform(a_group.iloc[:, target_index])

        for i, _doc in enumerate(a_group.iloc[:, target_index].tolist()):
            corpus.add_doc(_doc.strip().split())
            # if i % 1000 == 0:
            #     print('Document #{} has been loaded'.format(i))

        # If any element is larger than one, set it to one
        vector.data = np.where(vector.data > 0, 1, 0)
        time_stamp = a_group.iloc[:, date_index].tolist()[0]
        unique_time_stamp.append(time_stamp)
        burstvectors[time_stamp] = vector

        # print(time_stamp, time.time() - t0)

    with open('./models/unique_time_stamp.pickle', 'wb') as handle:
        pickle.dump(unique_time_stamp, handle)

    clusters = defaultdict(list)

    model = tm.LDAModel(k=30, min_cf=10, min_df=5, rm_top=50, corpus=corpus)
    model.optim_interval = 20
    model.burn_in = 200
    model.train(0)

    # Let's train the model
    for i in range(0, 1500, 20):
        # print('Iteration: {:04} LL per word: {:.4}'.format(i, model.ll_per_word))
        model.train(20)
    print('Iteration: {:04} LL per word: {:.4}'.format(1000, model.ll_per_word))

    # extract candidates for auto topic labeling
    extractor = tm.label.PMIExtractor(min_cf=10, min_df=10, max_len=5, max_cand=10000, normalized=True)
    cands = extractor.extract(model)
    labeler = tm.label.FoRelevance(model, cands, min_df=5, smoothing=1e-2, mu=0.25)

    cluster_label = {}
    for k in range(model.k):
        # print('Topic #{}'.format(k))
        m = 0
        for label, score in labeler.get_topic_labels(k, top_n=1):
            if m > 0:
                continue

            # print("Labels:", label)
            cluster_label[k] = label
            m += 1

        for word, prob in model.get_topic_words(k, top_n=20):
            # print('\t', word, prob, sep='\t')
            clusters[k].append(word)

    for key in sorted(clusters.keys()):
        print(key, ':', ', '.join(clusters[key]))
    # print(0, ':', ', '.join(clusters[0]))

    return model.k, clusters, cluster_label, bursts, burstvectors, unique_time_stamp

def plot_bersty_terms(output_filename, num_topics, clusters, cluster_label, bursts, burstvectors, unique_time_stamp):
    font_path = ''
    if platform.system() is 'Windows':
        # Window의 경우 폰트 경로
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    elif platform.system() is 'Darwin':
        # for Mac
        font_path = '/Library/Fonts/AppleGothic.ttf'

    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rc('axes', unicode_minus=False)

    y_val = int(np.ceil(num_topics / 4))

    # Graph selected bursty terms over time
    # yplots = 13
    # xplots = 4
    yplots = y_val
    xplots = 4
    fig, axs = plt.subplots(yplots, xplots)
    plt.subplots_adjust(right=1, hspace=0.9, wspace=0.3)
    plt.suptitle('Prevalence of selected bursty clusters over time', fontsize=14)
    fig.subplots_adjust(top=0.95)
    fig.set_figheight(16)
    fig.set_figwidth(12)

    # print(bursts[0])

    prevalences = []
    for i, cluster in enumerate(clusters):
        prevalence = get_prevalence(clusters[cluster], bursts, burstvectors, unique_time_stamp)

        x = range(0, len(prevalence))

        prevalences.append(prevalence)
        title = cluster_label[cluster]
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].plot(x, prevalence, color='k', ls='-', label=title)
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].grid()
        ymax = np.ceil(max(prevalence) * 2) / 2
        if ymax == 0.5 and max(prevalence) < 0.25:
            ymax = 0.25
        elif ymax == 2.5:
            ymax = 3
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].set_ylim(0, ymax)
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].set_xlim(0, len(prevalence))
        axs[int(np.floor((i / xplots) % yplots)), i % xplots].set_title(title, fontsize=12)

        if i % yplots != yplots - 1:
            axs[i % yplots, int(np.floor((i / yplots) % xplots))].set_xticklabels([])
        else:
            axs[i % yplots, int(np.floor((i / yplots) % xplots))].set_xticklabels([1988, 1998, 2008, 2018])

    axs[6, 0].set_ylabel('Percentage of documents containing term (%)', fontsize=12)

    # plt.show()
    plt.savefig(output_filename)
    plt.close(fig)
