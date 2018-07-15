
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from pprint import pprint
import sys, os
import numpy as np
import pandas as pd
import time
import calendar
import re
import unicodedata
import datetime
import math
import jpholiday


def extract_target(text):
    if ('待ちなし' in text) or ('待ち無' in text):
        return 0
    else:
        text = unicodedata.normalize('NFKC', text)
        s1 = re.compile('([0-9]+)人').search(text)
        s2 = re.compile('待ち([0-9]+)').search(text)
        s3 = re.compile('並び([0-9]+)').search(text)
        if s1:
            ninzu = int(s1.group(1))
        elif s2:
            ninzu = int(s2.group(1))
        elif s3:
            ninzu = int(s3.group(1))
        else:
            return -1

        if ninzu < 60:
            return ninzu
        else:
            return -1


def extract_time(year, month, day, hour, minute, week_num, num_of_day):
    daycycle_sin = math.sin(2 * math.pi * num_of_day / 365)
    daycycle_cos = math.cos(2 * math.pi * num_of_day / 365)
    week_sin = math.sin(2 * math.pi * week_num / 7)
    week_cos = math.cos(2 * math.pi * week_num / 7)
    today_is_holiday = 1 if jpholiday.is_holiday(datetime.date(year, month, day)) else 0
    tomorrow = datetime.datetime(year = year, month = month, day = day) + datetime.timedelta(days = 1)
    t_year = int(tomorrow.strftime('%Y'))
    t_month = int(tomorrow.strftime('%m'))
    t_day = int(tomorrow.strftime('%d'))
    tomorrow_is_holiday = 1 if jpholiday.is_holiday(datetime.date(t_year, t_month, t_day)) else 0
    min_of_day = ((hour - 10) * 60 + minute) / (12 * 60)
    return [daycycle_sin, daycycle_cos, week_sin, week_cos, today_is_holiday, tomorrow_is_holiday, min_of_day]


def extract_weather(year, month, day, weather_df):
    weather = ['晴', '曇', '雨', '雪', '雷', '風']
    weather_np = np.array(weather_df[weather_df['day'] == '{}/{}/{}'.format(year, month, day)])
    if weather_np.shape[0]:
        temp_h = [float(weather_np[0][1])]
        temp_l = [float(weather_np[0][2])]
        tenki = str(weather_np[0][4])
        tenki_list = [1 if t in tenki else 0 for t in weather]
        return temp_h + temp_l + tenki_list
    else:
        return -1


def make_feature_csv(tweet_csv_path='./data/tweet_data.csv',
                     X_csv_path='./data/X.csv',
                     Y_csv_path='./data/Y.csv',
                     Ymulti_csv_path='./data/Ymulti.csv',
                     weather_data_path='./data/weather_data.csv'):
                     
    weather_df = pd.read_csv(weather_data_path)
    tweet_df = pd.read_csv(tweet_csv_path, encoding='utf-8')
    clms = ['month_sin', 'month_cos', 'week_sin', 'week_cos','today_is_holiday', 'tomorrow_is_holiday',
            'minute_of_day', 'temp_h', 'temp_l', 'weat_sun', 'weat_cloud', 'weat_rain', 'weat_snow', 'weat_light', 'weat_storm']
    X = pd.DataFrame(columns=clms)
    Y = pd.DataFrame(columns=['target'])

    for key, tweet in tweet_df.iterrows():
        target = extract_target(tweet['text'])
        if target >= 0:
            Y = pd.concat([Y, pd.DataFrame({'target': [target]})], axis=0)
            new_df = pd.DataFrame([extract_time(int(tweet['year']),
                                                int(tweet['month']),
                                                int(tweet['day']),
                                                int(tweet['hour']),
                                                int(tweet['minute']),
                                                int(tweet['week_num']),
                                                int(tweet['num_of_day']))
                                   + extract_weather(str(tweet['year']),
                                                     str(tweet['month']),
                                                     str(tweet['day']),
                                                     weather_df)],
                                  columns=clms)
            X = pd.concat([X, new_df], axis=0)

    X.to_csv(X_csv_path, encoding='utf-8', index=False)
    Y.to_csv(Y_csv_path, encoding='utf-8', index=False)
    multi_clms = ['0_9', '10_19', '20_29', '30_39', '40_']
    multi_Y = np.array([[1,0,0,0,0] if n<10 else [0,1,0,0,0] if n<20 else [0,0,1,0,0] if n<30 else [0,0,0,1,0] if n<40 else [0,0,0,0,1] for n in np.array(Y)])
    multi_Y = pd.DataFrame(multi_Y, columns=multi_clms)
    multi_Y.to_csv(Ymulti_csv_path, encoding='utf-8', index=False)
    pprint('X shape : {0}'.format(X.shape))
    pprint('Y shape : {0}'.format(Y.shape))


def twitter_html_scrape(html_path='./data/webpage.html', save_path='./data/tweet_data.csv'):

    clms = ['text', 'year', 'month', 'day', 'hour', 'minute', 'week_num', 'num_of_day', 'num_of_week']
    tweets_df = pd.DataFrame(columns=clms)
    i = 0

    with open(html_path, 'r', encoding='utf-8') as fp:
        soup = BeautifulSoup(fp)

        for div_tag in soup.find_all('div'):
            try:
                if div_tag.get('class').pop(0) in 'content':
                    tw_time = ''
                    tw_text = ''

                    for elem in div_tag.find_all('span'):
                        try:
                            if elem.get('class').pop(0) in '_timestamp js-short-timestamp ':
                                tw_time = str(elem['data-time'])
                                break
                        except:
                            pass

                    for elem in div_tag.find_all('div'):
                        try:
                            if elem.get('class').pop(0) in 'js-tweet-text-container':
                                tw_text = str(elem.text)
                                break
                        except:
                            pass

                    if tw_time != '' and tw_text != '':
                        time_data = time.localtime(int(tw_time))
                        tw_text = tw_text.replace('\n', ' ')
                        if 9 < int(time.strftime("%H", time_data)) < 23 and int(time.strftime("%Y", time_data)) > 2011:
                            new_df = pd.DataFrame([[tw_text,
                                                    int(time.strftime("%Y", time_data)),
                                                    int(time.strftime("%m", time_data)),
                                                    int(time.strftime("%d", time_data)),
                                                    int(time.strftime("%H", time_data)),
                                                    int(time.strftime("%M", time_data)),
                                                    str(time.strftime("%w", time_data)),
                                                    int(time.strftime("%j", time_data)),
                                                    int(time.strftime("%U", time_data))]])
                            new_df.columns = clms
                            new_df.index = [i]
                            tweets_df = pd.concat([tweets_df, new_df], axis=0)
                            i += 1
            except:
                pass

    tweets_df.to_csv(save_path, encoding='utf-8', index=False)
    pprint('{0} tweets saved'.format(i))


if __name__ == '__main__':

    dirs = pd.read_json('dirs.json', typ='series')
    if not os.path.isdir(dirs['log_dir']):
        os.makedirs(dirs['log_dir'])
    sys.stdout = open(dirs['log_dir'] + 'preprocess.txt', 'w')

    twitter_html_scrape()
    make_feature_csv()

