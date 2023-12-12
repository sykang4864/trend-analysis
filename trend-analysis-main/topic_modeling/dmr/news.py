import numpy as np
import pandas as pd
import tomotopy as tp
from treform.topic_model.pyTextMinerTopicModel import pyTextMinerTopicModel

from _datasets import news_data
from topic_modeling.dmr.commons import dmr_model, topic_scoring, get_topic_labeler

pd.options.plotting.backend = "plotly"

# 생성 토픽 수
topic_number = 34
# 기존 생성한 모델 재사용여부
reuse_trained_model = False

model = None
if reuse_trained_model:
    # 모델 로드
    model = tp.DMRModel.load('./models/news.model')
else:
    # 데이터 로드 및 전처리
    timestamps, dataset = news_data.load_for_topic(timestamp_index=0, target_index=4, timestamp_pattern='%Y-%m', reuse_preproc=False)

    # DMR 모델 학습 및 저장

    print(type(timestamps[0]))
    print(timestamps)
    model = dmr_model(dataset, timestamps, topic_number)
    model.save('./models/news.model', True)

# document별 dominant topic 정보 저장
topic_model = pyTextMinerTopicModel()
df_topic_sents_keywords, matrix = topic_model.format_topics_sentences(topic_number=10, mdl=model)
# formatting
df_dominant_topic_for_each_doc = df_topic_sents_keywords.reset_index()
df_dominant_topic_for_each_doc.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic_for_each_doc.to_csv('./results/news_dominent_topic_for_each_doc.csv', index=False,
                                      encoding='utf-8-sig')

topicdocs = [0 for _ in range(model.k)]
for d in model.docs:
        topicdocs[d.get_topics(top_n=1)[0][0]] += 1

# 토픽별 label, keyword 저장
labeler = get_topic_labeler(model)
df_topic_label_keyword = pd.DataFrame(columns=['topic number', 'label', 'docs', 'keywords'])
for index, topic_number in enumerate(range(model.k)):
    label = ' '.join(label for label, score in labeler.get_topic_labels(topic_number, top_n=5))
    keywords = ' '.join(keyword for keyword, prob in model.get_topic_words(topic_number))
    df_topic_label_keyword.loc[index] = [topic_number, label, topicdocs[topic_number], keywords]
df_topic_label_keyword.to_csv('./results/news_topic_label_keyword.csv', index=False, encoding='utf-8-sig')

# timestamp별 topic score 계산 및 저장
df_topic_score = topic_scoring(model)
print(df_topic_score)
df_topic_score.to_csv('./results/news_topic_score.csv', index=False, encoding='utf-8-sig')

# timestamp별 topic score line graph
fig = df_topic_score.T.plot()
fig.update_layout(
    title="<b>Topics over Time</b>",
    template="plotly_white",
    xaxis_title="Timestamp",
    yaxis_title="Importance",
    legend_title="Topic",
    font=dict(
        size=18
    )
)
fig.write_html('./results/news_topic_over_time.html')

# timestamp별 topic distribution graph(using softmax)
probs = np.exp(model.lambdas - model.lambdas.max(axis=0))
probs /= probs.sum(axis=0)

df_probs = pd.DataFrame(data=probs).T
df_probs.index = model.metadata_dict
fig = df_probs.plot.barh()
fig.update_layout(
    title="<b>Topic Distributions</b>",
    template="plotly_white",
    xaxis_title="Distribution",
    yaxis_title="Timestamp",
    legend_title="Topic",
    font=dict(
        size=18
    )
)
fig.write_html('./results/news_topic_dist.html')
