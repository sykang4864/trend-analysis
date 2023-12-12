import requests
from pandas import json_normalize
import calendar
import pandas as pd

####################
# 금융감독원 보도자료 요약문 분석
# - Format : contentKor | atchfileUrl | subject | viewCnt | originUrl | contentId | atchfileNm | regDate | publishOrg
# - 기간 : 2017-2021
# - 분석대상 : contentKor
####################

auth_key = '810289f7039c2aac1394944305422bd4'

# 개인은 조회기간 1개월 제한
# 하루 조회 건수 30회
# for year in (2017, 2018, 2019, 2020, 2021):
for year in [2017, 2018, 2019, 2020, 2021]:
    for month in range(1, 13):
        month_range = calendar.monthrange(year, month)
        start_date = '-'.join([str(year), str(month).zfill(2), '01'])
        end_date = '-'.join([str(year), str(month).zfill(2), str(month_range[1]).zfill(2)])

        response = requests.get(
            'https://www.fss.or.kr/fss/kr/openApi/api/bodoInfo.jsp?apiType=json&startDate={}&endDate={}&authKey={}'.format(
                start_date, end_date, auth_key))
        try:
            result = response.json()['reponse']['result']
            print(response.json())
            dataset = json_normalize(result)
            dataset.to_csv('fss_{}.csv'.format(str(year) + str(month).zfill(2)), index=False, encoding='utf-8-sig')
        except:
            print(response.json())


li = []
for year in (2017, 2018, 2019, 2020, 2021):
    for month in range(1, 13):
        try:
            filename = 'fss_{}.csv'.format(str(year) + str(month).zfill(2))
            df_news = pd.read_csv("./" + str(filename), index_col=None, header=0)
            li.append(df_news)
        except:
            print('error : {}'.format(str(year) + str(month).zfill(2)))
            continue

# dataframe 통합
df_total = pd.concat(li, axis=0, ignore_index=True)
# nan 값을 갖는 row 제거
df_total.dropna(axis=0, inplace=True)
# date 기준 오름차순 정렬
df_total.sort_values(by=['regDate'], axis=0, inplace=True)
# csv 출력
df_total.to_csv('fss.csv', index=False, encoding='utf-8-sig')
