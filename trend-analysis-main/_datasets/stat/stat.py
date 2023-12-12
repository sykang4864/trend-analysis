import pandas as pd
import plotly.express as px
from nltk import word_tokenize, sent_tokenize
import re

pd.options.plotting.backend = "plotly"

def plot_doc_count(df_jpss, df_kpsa, df_news):
    df_jpss_cnt = df_jpss.groupby(['year'])['year'].count()
    df_kpsa_cnt = df_kpsa.groupby(['year'])['year'].count()
    df_news_cnt = df_news.groupby(df_news.date.dt.year)['date'].count()

    df_cnt = pd.concat([df_jpss_cnt, df_kpsa_cnt, df_news_cnt], axis=1)
    df_cnt.columns = ['JPSS', 'KPSA', 'News']
    df_cnt.index = df_cnt.index.astype(int)
    df_cnt.to_csv('./stat_doc_count.csv', encoding='utf-8-sig')

    fig = df_cnt.plot(kind='bar', text_auto=True)
    fig.update_layout(
        title="<b>Number of documents in each media by year</b>",
        template="plotly_white",
        xaxis_title="Year",
        yaxis_title="Number of documents",
        legend_title="Media",
        font=dict(
            size=18
        ),
        barmode='group'
    )
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.write_html('./stat_doc_count.html')


def plot_word_count(df_jpss, df_kpsa, df_news):
    df_jpss = df_jpss[['year', 'abstract']]
    df_kpsa = df_kpsa[['year', 'abstract']]
    df_news = df_news[['date', 'content']]

    # jpss 연도별 단어 수
    num_words = []
    for document in df_jpss['abstract']:
        document = re.sub('[^A-Za-z0-9가-힣_ ]+', '', document)
        words = word_tokenize(document)
        num_words.append(len(words))
    df_jpss['num_words'] = num_words
    df_jpss_cnt = df_jpss.groupby(['year'])['num_words'].sum()

    # kpsa 연도별 단어 수
    num_words = []
    for document in df_kpsa['abstract']:
        document = re.sub('[^A-Za-z0-9가-힣_ ]+', '', document)
        words = word_tokenize(document)
        num_words.append(len(words))
    df_kpsa['num_words'] = num_words
    df_kpsa_cnt = df_kpsa.groupby(['year'])['num_words'].sum()

    num_words = []
    for document in df_news['content']:
        document = re.sub('[^A-Za-z0-9가-힣_ ]+', '', str(document))
        words = word_tokenize(document)
        num_words.append(len(words))
    df_news['num_words'] = num_words
    df_news_cnt = df_news.groupby(df_news.date.dt.year)['num_words'].sum()

    df_cnt = pd.concat([df_jpss_cnt, df_kpsa_cnt, df_news_cnt], axis=1)
    df_cnt.columns = ['JPSS', 'KPSA', 'News']
    df_cnt.index = df_cnt.index.astype(int)
    df_cnt.to_csv('./stat_word_count.csv', index=False, encoding='utf-8-sig')
    print(df_cnt)

    fig = df_cnt.plot(kind='bar', text_auto=True)
    fig.update_layout(
        title="<b>Number of words in each media by year</b>",
        template="plotly_white",
        xaxis_title="Year",
        yaxis_title="Number of words",
        legend_title="Media",
        font=dict(
            size=18
        ),
        barmode='group'
    )
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.write_html('./stat_word_count.html')

df_jpss = pd.read_csv('../jpss.csv')
df_kpsa = pd.read_csv('../kpsa.csv')
df_news = pd.read_csv('../news.csv')
df_news['date'] = pd.to_datetime(df_news['date'])

# 연도별 각 매체의 문서 수
plot_doc_count(df_jpss, df_kpsa, df_news)

# 연도별 각 매체의 단어 수
plot_word_count(df_jpss, df_kpsa, df_news)
