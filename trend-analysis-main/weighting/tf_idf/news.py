from _datasets import news_data
import treform as ptm
import re
import pandas as pd

####################
# 네이버 뉴스
# - Format : date | press | title | link | content
# - 기간 : 2017-2021
# - 총 5257건
# - 분석대상 : content
####################

# 데이터 로드 및 전처리
label, content = news_data.load_for_term_weighting(label_index=2, target_index=4)

documents = []
for index, doc in enumerate(content):
    document = ' '
    for sent in doc:
        for word in sent:
            new_word = re.sub('[^A-Za-z0-9가-힣_ ]+', '', word)
            if len(new_word) > 0:
                document += ' ' + word
    document = document.strip()
    if len(document) > 0:
        documents.append(document)
    else:
        print('remove {}th content '.format(str(index)))
        label.pop(index)   # label 제거

print('label length : ' + str(len(label)))
print('content length : ' + str(len(content)))

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
df_results[:100].to_csv('./results/news_tf_idf.csv', index=False, encoding='utf-8-sig')
