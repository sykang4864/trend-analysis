import pandas as pd
import treform as ptm

from _datasets import jpss_data

####################
# Journal of Payments Strategy & Systems 논문
# - Format : id(eid) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

# 데이터 로드 및 전처리
label, documents = jpss_data.load_for_term_weighting(label_index=0, target_index=3)

# Term weighting : TF-IDF
tf_idf = ptm.weighting.TfIdf(documents, label_list=label)
weights = tf_idf()

# Term weighting 값 기준 정렬 및 csv 저장
label_term_value_list = []
for label, term_value in weights.items():
    for term, value in term_value.items():
        label_term_value_list.append((label, term, value))
        print('{}\t{}\t{}'.format(label, term, str(value)))
df_results = pd.DataFrame(label_term_value_list, columns=['label', 'term', 'tf-idf'])
df_results.sort_values(by=['tf-idf'], ascending=False, inplace=True)
df_results[:100].to_csv('./results/jpss_tf_idf.csv', index=False, encoding='utf-8-sig')
