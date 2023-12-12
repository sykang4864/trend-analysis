import treform.keyword.textrank as tr

from _datasets import news_data
import spacy
import pytextrank

####################
# 네이버 뉴스
# - Format : date | press | title | link | content
# - 기간 : 2017-2021
# - 총 8119건
# - 분석대상 : content
####################

# 데이터 로드 및 전처리
doc_group_by_time = news_data.load_for_keyword(timestamp_name='date', target_name='content',
                                               timestamp_format='%Y-%m-%d',
                                               timestamp_group_format='Y', reuse_preproc=True)

# TextRank 기반 Keyword 추출
for timestamp in doc_group_by_time.keys():
    # print(timestamp)
    # documents = doc_group_by_time[timestamp]
    # # python -m spacy download ko_core_news_lg or ko_core_news_sm
    # # https://github.com/DerwenAI/pytextrank
    # # https://spacy.io/models/ko#ko_core_news_lg
    # nlp = spacy.load("ko_core_news_sm", disable=["morphologizer", "attribute_ruler", "ner", "lemmatizer"])
    # nlp.add_pipe("textrank")
    #
    # print(nlp.pipe_names)
    #
    # doc = nlp(' '.join(documents))
    #
    # # examine the top-ranked phrases in the document
    # for rank in doc._.textrank.calc_textrank():
    #     print(rank)
    #
    # # for phrase in doc._.phrase[:10]:
    #     # print(phrase.rank, phrase.count)
    #     # print(phrase.chunks)

    print(timestamp)
    documents = doc_group_by_time[timestamp]
    keyword_extractor = tr.TextRank(pos_tagger_name='mecab', mecab_path='/home/sklim/mecab-ko-dic-2.1.1-20180720/', lang='ko')
    keyword_extractor.build_keywords(' '.join(documents))
    keywords = keyword_extractor.get_keywords(limit=10)

    with open('./results/{}_news_keywords.txt'.format(timestamp.strftime('%Y')), 'w', encoding='utf-8') as fout:
        for word, r in keywords:
            print('{}\t{}\n'.format(word, r))
            fout.write('{}\t{}\n'.format(word, r))
    fout.close()
