# https://github.com/MaartenGr/BERTopic

from bertopic import BERTopic
from _datasets import news_data
import pandas as pd

# 데이터 로드
timestamps, dataset = news_data.load_for_bertopic(timestamp_index=0, target_index=4, timestamp_pattern='%Y%m', reuse_preproc=False)

# 기존 생성한 모델 재사용여부
reuse_trained_model = False
if reuse_trained_model:
    topic_model = BERTopic.load('./models/news.model')
    topics, probs = topic_model.transform(dataset)
else:
    topic_model = BERTopic(language='multilingual')  # select a model that supports 50+ languages.
    topics, probs = topic_model.fit_transform(dataset)
    topic_model.save('./models/news.model')

# topic name, count, keyword 저장
df_topic_info = topic_model.get_topic_info()

dict_topics = topic_model.get_topics()
for topic_id in dict_topics:
    topic = dict_topics[topic_id]
    # [(keyword1, probs1), ......] (list(zip))→ [keyword1, ...] (join)→ keyword1 keyword2 ...
    keywords = ' '.join(list(zip(*topic))[0])
    df_topic_info.loc[df_topic_info['Topic'] == topic_id, 'Keywords'] = keywords
print(df_topic_info)
df_topic_info.to_csv('./results/news_topic_info.csv', encoding='utf-8-sig')

# topic keyword score bar chart
fig = topic_model.visualize_barchart()
fig.write_html("./results/news_topic_keywords_score.html")

# topic similarity heatmap
fig = topic_model.visualize_heatmap(top_n_topics=10)
fig.write_html("./results/news_topic_similarity_heatmap.html")

# dynamic topic modeling (over time)
topics_over_time = topic_model.topics_over_time(dataset, topics, timestamps, datetime_format='%Y%m', nr_bins=20)
topics_over_time.to_csv('./results/news_topic_over_time.csv')
fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
fig.write_html("./results/news_topic_over_time.html")
