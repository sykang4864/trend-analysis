## Framework for Trend Analysis
The framework for trend analysis isn't restricted solely to specific topics such as digital technology, fintech, payment, and settlement. It can be applied to any topic for which you have textual data and wish to conduct analysis.



--------------------------------------------------------------------------------
**Architecture**

① Input the collected textual data like news and journals which you want to anlayze
② Pre-processing into a format available to anlyze
Conduct ③-(1)word ranking, ③-(2)co-occurrence counting , ③-(3)topic modeling
As a result of ③, conduct ④-(1) keyword analysis, ④-(2) keyword network analysis, ④-(3) time-series topic analysis



--------------------------------------------------------------------------------

**Experiment for Trend Analysis**

Using the framework for trend analysis, a study was conducted on the trend analysis about 'payment systems.' 
Datasets : the Journal of Payments Strategy & Systems (JPSS), the Korean Payment and Settlement Association Journal (KPSA), and articles from Naver News

|classfication|Collection|Analysis|
|------|---|---|
|International Journal|[Journal of Payments Strategy & Systems(JPSS)](https://www.henrystewartpublications.com/jpss)|164 abstracts|
|Domestic Journal|[the Korean Payment and Settlement Association Journal(KPSA)](http://www.kpsa.kr/)|93 abstracts|
|News|[News from Naver Portal](https://news.naver.com/)-serach by ‘Payment’|8119 contents|


--------------------------------------------------------------------------------

**code example**

```sh
$ python kpsa.py
```
