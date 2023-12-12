# 크롤링시 필요한 라이브러리 불러오기
from bs4 import BeautifulSoup
import requests
import re
from selenium import webdriver
import pandas as pd

# 웹드라이버 설정
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)

def find_naver_news_url(query, date_start, date_end):
    news_url = 'https://search.naver.com/search.naver?where=news&query={}&sm=tab_opt&sort=0&photo=0&field=0&pd=3&ds={}&de={}'

    req = requests.get(news_url.format(query, date_start, date_end))
    soup = BeautifulSoup(req.text, 'html.parser')

    idx = 0
    cur_page = 1

    news_num = 4000
    print('크롤링 중...')

    naver_urls = []

    while idx < news_num:
        ### 네이버 뉴스 웹페이지 구성이 바뀌어 태그명, class 속성 값 등을 수정함(20210126) ###

        table = soup.find('ul', {'class': 'list_news'})  # table : 뉴스 바운딩 박스(ul 태그)
        li_list = table.find_all('li', {'id': re.compile('sp_nws.*')})  # li_list : 뉴스 바운딩 박스 안의 각 뉴스 기사(li 태그)
        area_list = [li.find('div', {'class': 'news_area'}) for li in
                     li_list]  # area_list : 뉴스 기사 안의 뉴스 제목, 본문이 담긴 태그(div 태그)
        a_list_of_list = [area.select('a.info') for area in
                  area_list]  # a_list : 각 뉴스기사 내부 title, URL 정보가 담긴 태그(a 태그)

        for a_list in a_list_of_list:
            for a_info in a_list:
                url = a_info.get('href')
                if "news.naver.com" in url:
                    if not (url in naver_urls):
                        print(url)
                        naver_urls.append(url)
                        idx += 1

        print('current page : ' + str(cur_page))
        if cur_page == 400:
            break
        cur_page += 1

        pages = soup.find('div', {'class': 'sc_page_inner'})
        try:
            next_page_url = [p for p in pages.find_all('a') if p.text == str(cur_page)][0].get('href')
        except:
            print(pages)
            break

        req = requests.get('https://search.naver.com/search.naver' + next_page_url)
        soup = BeautifulSoup(req.text, 'html.parser')

    print('URL 크롤링 완료')

    return naver_urls

## selenium으로 navernews만 뽑아오기##

query = '지급결제'
date_start = '2018.07.01'   # 2017.01.01, 2017.07.01, 2018.01.01, 2018.07.01
                            # 2019.01.01, 2019.07.01, 2020.01.01, 2020.07.01, 2021.01.01, 2021.07.01
date_end = '2018.12.31'     # 2017.06.30, 2017.12.31, 2018.06.30, 2018.12.31,
                            # 2019.06.30, 2019.12.31, 2020.06.30, 2020.12.31, 2021.06.30, 2021.12.31
naver_urls = find_naver_news_url(query, date_start, date_end)
print('네이버 뉴스 : {}개'.format(len(naver_urls)))

###naver 기사 본문 및 제목 가져오기###

# ConnectionError방지
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}

date_list = []
press_list = []
titles = []
contents = []

# naver_urls = []
# naver_urls.append('https://news.naver.com/main/read.naver?mode=LSD&mid=sec&sid1=101&oid=003&aid=0008681710')

for index, url in enumerate(naver_urls):
    original_html = requests.get(url, headers=headers)
    html = BeautifulSoup(original_html.text, "html.parser")

    # 뉴스 제목 가져오기
    # title = html.select("div.content > div.article_header > div.article_info > h3")
    title = html.select("div.media_end_head_title > h2" )

    # list합치기
    title = ''.join(str(title))
    # html태그제거
    pattern1 = '<[^>]*>'
    title = re.sub(pattern=pattern1, repl='', string=title)
    titles.append(title)
    print(title)

    # 시간
    date_time = html.find('div', {'class': 'media_end_head_info_datestamp_bunch'})
    if date_time != None:
        date_time = date_time.find('span', {'class': 'media_end_head_info_datestamp_time _ARTICLE_DATE_TIME'})
        if len(date_time) != 0:
            date_time = re.sub(pattern=pattern1, repl='', string=str(date_time))
            print(date_time)
            date_list.append(date_time)
        else:
            date_list.append('')
    else:
        date_list.append('')


    # 언론사
    # press = html.find('a', {'class': 'nclicks(atp_press)'})
    press = html.find('a', {'class': 'media_end_head_top_logo'})
    if press != None:
        # press_list.append(press.find('img').get('title'))
        print(press.find('img').get('title'))
        press_list.append(press.find('img').get('title'))
    else:
        press_list.append('')

    # 뉴스 본문 가져오기
    # content = html.select("div.content > div#articleBody > div#articleBodyContents")
    content = html.find('div', {'class': 'go_trans _article_content'})

    # 기사 텍스트만 가져오기
    # list합치기
    content = ''.join(str(content))

    # html태그제거 및 텍스트 다듬기
    content = re.sub(pattern=pattern1, repl='', string=content)
    pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
    content = content.replace(pattern2, '')
    content = content.replace('\n', '')

    contents.append(content)

    print(content)

    if (index+1) % 100 == 0:
        print('본문 크롤링 완료 : {}개'.format(str(index+1)))


# 데이터프레임으로 정리(titles,url,contents)
news_df = pd.DataFrame(
    {'date': date_list, 'press': press_list, 'title': titles, 'link': naver_urls, 'content': contents})

news_df.to_csv('news_{}_{}.csv'.format(date_start, date_end), index=False, encoding='utf-8-sig')

# import pandas as pd
#
# # filenames = ['news_2019.01.01_2019.06.30.csv', 'news_2019.07.01_2019.12.31.csv', 'news_2020.01.01_2020.06.30.csv',
# #              'news_2020.07.01_2020.12.31.csv', 'news_2021.01.01_2021.06.30.csv', 'news_2021.07.01_2021.12.31.csv']
#
# filenames = ['news_2017.01.01_2017.06.30.csv', 'news_2017.07.01_2017.12.31.csv', 'news_2018.01.01_2018.06.30.csv',
#              'news_2018.07.01_2018.12.31.csv']
#
# li = []
# for i in range(len(filenames)):
#     print(filenames[i])
#     df_news = pd.read_csv("./" + str(filenames[i]), index_col=None, header=0)
#     li.append(df_news)
# # dataframe 통합
# df_total = pd.concat(li, axis=0, ignore_index=True)
#
# # nan 값을 갖는 row 제거
# df_total.dropna(axis=0, inplace=True)
# # date 기준 오름차순 정렬
# df_total.sort_values(by=['date'], axis=0, inplace=True)
# # csv 출력
# df_total.to_csv('news.csv', index=False, encoding='utf-8-sig')