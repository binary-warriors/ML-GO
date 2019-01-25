from newsapi import NewsApiClient
import datetime
from dateutil.relativedelta import relativedelta
import os
from random import shuffle

proxy = 'http://edcguest:edcguest@172.31.100.14:3128'

os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

newsapi = NewsApiClient(api_key='4dbc17e007ab436fb66416009dfb59a8')



today = datetime.datetime.now()
one_month_before = datetime.datetime.now() - relativedelta(months=1)
date_formated_today = today.strftime("%Y-%m-%d")
date_formated_one_month_before = one_month_before.strftime("%Y-%m-%d")

print(date_formated_today, date_formated_one_month_before)

all_articles = newsapi.get_everything(q='machine learning',
                                      sources='bbc-news,google-news,google-news-in,time,cnn,the-verge,financial-times',
                                      from_param=date_formated_one_month_before,
                                      to=date_formated_today,
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)

# sources = newsapi.get_sources()

# print(all_articles)

c = 1
articles = all_articles['articles']
result = []
for article in articles:
    print('===============================================================================')
    t_dict = {}
    for k, v in article.items():
        # print(c, k, v)
        if k == 'source':
            for key, val in v.items():
                # print(key, val)
                t_dict[key] = val
        else:
            t_dict[k] = v

    result.append(t_dict)
    c += 1

print(result)
