from krwordrank.word import KRWordRank


def execute_KRWordRank(documents, min_count=5, max_length=10, beta=0.85, max_iter=10, num_words=100):
    krword_rank = KRWordRank(min_count, max_length, verbose=True)
    keywords, rank, graph = krword_rank.extract(docs=documents, beta=beta, max_iter=max_iter, num_keywords=num_words)

    return keywords
