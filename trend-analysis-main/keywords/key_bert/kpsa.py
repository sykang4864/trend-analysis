from keybert import KeyBERT

from _datasets import kpsa_data

####################
# 지급결제학회 논문
# - Format : id(doi) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

# 데이터 로드 및 전처리
doc_group_by_time = kpsa_data.load_for_keyword(timestamp_name='year', target_name='abstract', timestamp_format='%Y',
                                               timestamp_group_format='Y', reuse_preproc=True)

# KeyBERT 기반 Keyword 추출
for timestamp in doc_group_by_time.keys():
    documents = doc_group_by_time[timestamp]
    model = KeyBERT('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    keywords = model.extract_keywords(' '.join(documents), top_n=10, keyphrase_ngram_range=(1, 1))

    with open('./results/{}_kpsa_keywords.txt'.format(timestamp.strftime('%Y')), 'w', encoding='utf-8') as fout:
        for word, r in keywords:
            print('{}\t{}\n'.format(word, r))
            fout.write('{}\t{}\n'.format(word, r))
    fout.close()
