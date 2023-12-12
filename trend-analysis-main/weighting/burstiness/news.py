from weighting.burstiness.commons import compute_term_burstiness
from weighting.burstiness.commons import topic_modeling
from weighting.burstiness.commons import plot_bersty_terms
from _datasets import news_data
import pandas as pd
import os.path

# 데이터 로드
df = news_data.load_for_term_burstiness(target_index=5, ngram_min=1, ngram_max=1)

# CountVectorizer
# - time stamp = document
# - When building the vocabulary ignore terms that have a document
#   frequency strictly lower than the given threshold.
min_time_stamp_df = 3

short_ma_length = 6  # 단기 이동평균선
long_ma_length = 12  # 장기 이동평균선
signal_line_ma = 3  # 시그널 곡선 : N일 동안의 MACD 지수 이동평균
significance_ma_length = 3  #

# term burstiness 계산
burstiness, burstiness_over_time = compute_term_burstiness(df=df,
                                                           date_index=0,
                                                           target_index=5,
                                                           min_time_stamp_df = min_time_stamp_df,
                                                           long_ma_length=long_ma_length,
                                                           short_ma_length=short_ma_length,
                                                           significance_ma_length=significance_ma_length,
                                                           signal_line_ma=signal_line_ma)

# 시간별 term burstiness 계산 결과 csv 저장
burstiness_over_time.to_csv(
    './results/news_bursty_terms_over_time_{}_{}_{}_{}.csv'.format(short_ma_length, long_ma_length, signal_line_ma,
                                                                   significance_ma_length),
    encoding='utf-8-sig')
# term bustiness 계산 결과 csv 저장
# - 결과 형식 : terms | max value | location(month)
burstiness.to_csv('./results/news_bursty_terms_{}_{}_{}_{}.csv'.format(short_ma_length, long_ma_length, signal_line_ma,
                                                                       significance_ma_length),
                  encoding='utf-8-sig')

# topic modeling
num_topics, clusters, cluster_label, bursts, burstvectors, unique_time_stamp \
    = topic_modeling(burstiness, df=df, date_index=0, target_index=5)

# save topics
df_clusters = pd.DataFrame(columns=['topic number', 'label', 'keywords'])
for index, key in enumerate(sorted(clusters.keys())):
    words = ' '.join(clusters[key])
    label = cluster_label[key]
    df_clusters.loc[index] = [key, label, words]
df_clusters.to_csv(
    './results/news_bursty_clusters_{}_{}_{}_{}.csv'.format(short_ma_length, long_ma_length, signal_line_ma,
                                                            significance_ma_length), index=False, encoding='utf-8-sig')

# plot clusters
output_filename = './results/news_bursty_clusters_{}_{}_{}_{}.png'.format(short_ma_length, long_ma_length,
                                                                          signal_line_ma, significance_ma_length)
plot_bersty_terms(output_filename, num_topics, clusters, cluster_label, bursts, burstvectors, unique_time_stamp)

# bursty term 기반 문장 생성
# https://main-ko-gpt2-scy6500.endpoint.ainize.ai/?utm_medium=social&utm_source=velog&utm_campaign=everyone%20ai&utm_content=kogpt2

# for short_ma_length in range(2, 9):  # 2~8
#     for long_ma_length in range(9, 19):  # 9~18
#         for signal_line_ma in range(2, 7):  # 2~6
#             for significance_ma_length in range(2, 7):  # 2~6
#                 try:
#                     # for restart
#                     filename = './results/news_bursty_clusters_{}_{}_{}_{}.png'.format(short_ma_length, long_ma_length,
#                                                                                        signal_line_ma,
#                                                                                        significance_ma_length)
#                     if os.path.isfile(filename):
#                         continue
#
#                     print('##### start {}/{}/{}/{}'.format(short_ma_length, long_ma_length,
#                                                            signal_line_ma, significance_ma_length))
#                     burstiness = compute_term_burstiness(df=df,
#                                                          date_index=0,
#                                                          target_index=5,
#                                                          long_ma_length=long_ma_length,
#                                                          short_ma_length=short_ma_length,
#                                                          significance_ma_length=significance_ma_length,
#                                                          signal_line_ma=signal_line_ma)
#                     # term bustiness 계산 결과 csv 저장
#                     # - 결과 형식 : terms | max value | location(month)
#                     filename = './results/news_bursty_terms_{}_{}_{}_{}.csv'.format(short_ma_length, long_ma_length,
#                                                                                     signal_line_ma,
#                                                                                     significance_ma_length)
#                     burstiness.to_csv(filename, encoding='utf-8-sig')
#
#                     # topic modeling
#                     num_topics, clusters, cluster_label, bursts, burstvectors, unique_time_stamp \
#                         = topic_modeling(burstiness, df=df, date_index=0, target_index=5)
#
#                     # save topics
#                     df_clusters = pd.DataFrame(columns=['key', 'label', 'words'])
#                     for i, key in enumerate(sorted(clusters.keys())):
#                         words = ', '.join(clusters[key])
#                         label = cluster_label[key]
#                         df_clusters.loc[i] = [key, label, words]
#                     df_clusters.to_csv(
#                         './results/news_bursty_clusters_{}_{}_{}_{}.csv'.format(short_ma_length, long_ma_length,
#                                                                                 signal_line_ma, significance_ma_length),
#                         index=False,
#                         encoding='utf-8-sig')
#
#                     # plot clusters
#                     output_filename = './results/news_bursty_clusters_{}_{}_{}_{}.png'.format(short_ma_length,
#                                                                                               long_ma_length,
#                                                                                               signal_line_ma,
#                                                                                               significance_ma_length)
#                     plot_bersty_terms(output_filename, num_topics, clusters, cluster_label, bursts, burstvectors,
#                                       unique_time_stamp)
#                 except:
#                     print('error!')
#                     ferr = open('./results/error.txt', 'a')
#                     ferr.write(
#                         'short: {}\tlong: {}\tsignal: {}\tsignificance: {}\n'.format(short_ma_length, long_ma_length,
#                                                                                      signal_line_ma,
#                                                                                      significance_ma_length))
#                     print('short: {}, long: {}, signal: {}, significance: {}'.format(short_ma_length, long_ma_length,
#                                                                                      signal_line_ma,
#                                                                                      significance_ma_length))
#                     ferr.close()
