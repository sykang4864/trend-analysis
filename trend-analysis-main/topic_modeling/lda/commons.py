import tomotopy as tp
import sys
import pandas as pd
import numpy as np


# LDA 모델 생성 및 학습
def lda_model(documents, topic_number, min_cf=3, rm_top=5, iter=1500):
    model = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=min_cf, rm_top=rm_top, k=topic_number)
    for document in documents:
        model.add_doc(document)
    model.burn_in = 100

    model.train(0)
    print('Num docs:', len(model.docs), ', Vocab size:', model.num_vocabs, ', Num words:', model.num_words)
    print('Removed top words:', model.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    for i in range(0, iter, 10):
        model.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, model.ll_per_word))

    print('Perplexity: {}'.format(model.perplexity))

    return model


# topic labeler 생성
def get_topic_labeler(model):
    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    cands = extractor.extract(model)
    labeler = tp.label.FoRelevance(model, cands, min_df=5, smoothing=1e-2, mu=0.25)
    return labeler
