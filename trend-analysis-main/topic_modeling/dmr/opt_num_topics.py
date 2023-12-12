import os

import pandas as pd
import plotly.graph_objects as go
import tomotopy as tp
from plotly.subplots import make_subplots

from _datasets import news_data
from topic_modeling.dmr.commons import dmr_model

# Topic 수 별 LDA 모델 생성
part = 'News'
num_topics_list = list(range(2, 101, 2))
if os.path.exists(f'./models/{part}_num_topics_2.model') == False:
    timestamps, dataset = news_data.load_for_topic(timestamp_index=0, target_index=4, reuse_preproc=True)
    for num_topics in num_topics_list:
        print('##### number of topics: {}'.format(num_topics))
        model = dmr_model(dataset, timestamps, num_topics)
        model.save(f'./models/{part}_num_topics_{num_topics}.model', full=True)

# perplexity/coherence 계산
coherence_metric = 'c_v'  # u_mass(0에 가까울수록 일관성 높음), c_uci, c_npmi, c_v(0과1사이, 0.55정도 수준)
perplexities = []
coherences = []
for num_topics in num_topics_list:
    model_name = './models/' + part + '_num_topics_' + str(num_topics) + '.model'
    model = tp.DMRModel.load(model_name)
    coherence = tp.coherence.Coherence(model, coherence=coherence_metric)
    print('num topics: {}\tperplexity: {}\tcoherence: {}'.format(num_topics, model.perplexity, coherence.get_score()))
    perplexities.append(model.perplexity)
    coherences.append(coherence.get_score())

# csv 저장
columns = ['topic number', 'perplexity', 'coherence', 'metric']
df_result = pd.DataFrame(columns=columns)
df_result = df_result.append(
    pd.DataFrame((zip(num_topics_list, perplexities, coherences, [coherence_metric] * len(num_topics_list))),
                 columns=columns), ignore_index=True)
print(df_result)

result_file = f'./results/{part}_perplexity_coherence.csv'
df_result.to_csv(result_file, index=False, encoding='utf-8-sig')

# plot perplexity/coherence
fig = make_subplots(specs=[[{"secondary_y": True}]])
trace1 = go.Scatter(x=df_result['topic number'], y=df_result['perplexity'], name='Perplexity')
trace2 = go.Scatter(x=df_result['topic number'], y=df_result['coherence'], name='Coherence')
fig.add_trace(trace1, secondary_y=False)
fig.add_trace(trace2, secondary_y=True)
fig.update_layout(title_text=part + ' Perplexity and Coherence')
fig.update_yaxes(title_text='Perplexity', secondary_y=False)
fig.update_yaxes(title_text='Coherence', secondary_y=True)
fig.update_xaxes(title_text='Number of topics')
fig.write_html(os.path.splitext(result_file)[0] + '.html')
