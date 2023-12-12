from korea_news_crawler.articlecrawler import ArticleCrawler

Crawler  = ArticleCrawler()
Crawler.set_category('경제','IT과학')
Crawler.set_date_range("2022-4-20", "2022-04-20")
Crawler.start()
