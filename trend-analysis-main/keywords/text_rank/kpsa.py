import treform.keyword.textrank as tr

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

# TextRank 기반 Keyword 추출
for timestamp in doc_group_by_time.keys():
    documents = doc_group_by_time[timestamp]
    keyword_extractor = tr.TextRank(pos_tagger_name='mecab', mecab_path='/home/sklim/mecab-ko-dic-2.1.1-20180720/', lang='ko')
    keyword_extractor.build_keywords(' '.join(documents))
    keywords = keyword_extractor.get_keywords(limit=10)

    with open('./results/{}_kpsa_keywords.txt'.format(timestamp.strftime('%Y')), 'w', encoding='utf-8') as fout:
        for word, r in keywords:
            print('{}\t{}\n'.format(word, r))
            fout.write('{}\t{}\n'.format(word, r))
    fout.close()
