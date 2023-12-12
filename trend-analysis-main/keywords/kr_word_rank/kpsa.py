from _datasets import kpsa_data
from keywords.kr_word_rank.commons import execute_KRWordRank

####################
# 지급결제학회 논문
# - Format : id(doi) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

# 데이터 로드 및 전처리
doc_group_by_time = kpsa_data.load_for_keyword(timestamp_name='year', target_name='abstract', timestamp_format='%Y',
                                               timestamp_group_format='Y', reuse_preproc=True)

####################
# KRWordRank 기반 Keyword 추출
# - 파라미터: GitHub에 공유된 KRWordRank 테스트 코드의 기본 설정값
# - GitHub: https://github.com/lovit/KR-WordRank/blob/master/tests/test_krwordrank.py
####################
for timestamp in doc_group_by_time.keys():
    documents = doc_group_by_time[timestamp]
    keywords = execute_KRWordRank(documents,
                                  min_count=5,  # 단어의 최소 출현 빈도수 (그래프 생성 시)
                                  max_length=10,  # 단어의 최대 길이
                                  beta=0.85,  # PageRank의 decaying factor beta
                                  max_iter=10,
                                  num_words=10)

    with open('./results/{}_kpsa_keywords.txt'.format(timestamp.strftime('%Y')), 'w', encoding='utf-8') as fout:
        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
            print('{}\t{}\n'.format(word, r))
            fout.write('{}\t{}\n'.format(word, r))
    fout.close()
