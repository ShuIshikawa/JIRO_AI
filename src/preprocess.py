
import pandas as pd
import numpy as np
import re
import unicodedata
import datetime
from sklearn.model_selection import train_test_split

import holiday_list
holidays = holiday_list.holiday

weather = ['晴', '曇', '雨', '雪', '雷', '風']
weather_df = pd.read_csv('./resources/weather_data.csv')

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

def extract_time(year, month, day, hour, minute, week_num):
    month_list = [1 if n == month - 1 else 0 for n in range(12)]
    week_list = [1 if n == week_num % 7 else 0 for n in range(7)]
    today_is_holiday = [1 if '{}/{}/{}'.format(year, month, day) in holidays else 0]
    tomorrow = datetime.datetime(year = year, month = month, day = day) + datetime.timedelta(days = 1)
    t_year = int(tomorrow.strftime('%Y'))
    t_month = int(tomorrow.strftime('%m'))
    t_day = int(tomorrow.strftime('%d'))
    tomorrow_is_holiday = [1 if '{}/{}/{}'.format(t_year, t_month, t_day) in holidays else 0]
    min_of_day = [((hour - 10) * 60 + minute) / (12 * 60)]
    return month_list + week_list + today_is_holiday + tomorrow_is_holiday + min_of_day

def extract_weather(year, month, day):
    weather_np = np.array(weather_df[weather_df['day']=='{}/{}/{}'.format(year, month, day)])
    if weather_np.shape[0]:
        temp_h = [float(weather_np[0][1])]
        temp_l = [float(weather_np[0][2])]
        tenki = str(weather_np[0][4])
        tenki_list = [1 if t in tenki else 0 for t in weather]
        return temp_h + temp_l + tenki_list
    else:
        return -1

def main():
    tweet_df = pd.read_csv('./resources/tweet_data.csv')

    clms = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
            'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat',
            'today_is_holiday', 'tomorrow_is_holiday',
            'minute_of_day',
            'temp_h', 'temp_l',
            'weat_sun', 'weat_cloud', 'weat_rain', 'weat_snow', 'weat_light', 'weat_storm']

    X = pd.DataFrame(columns=clms)
    Y = pd.DataFrame(columns=['target'])

    j = 0
    for key, tweet in tweet_df.iterrows():
        target = extract_target(tweet['text'])
        if target >= 0:

            Y = pd.concat([Y, pd.DataFrame({'target': [target]})], axis=0)
            new_df = pd.DataFrame([extract_time(int(tweet['year']),
                                                int(tweet['month']),
                                                int(tweet['day']),
                                                int(tweet['hour']),
                                                int(tweet['minute']),
                                                int(tweet['week_num']),)
                                   + extract_weather(str(tweet['year']),
                                                     str(tweet['month']),
                                                     str(tweet['day']))],
                                  columns=clms)
            X = pd.concat([X, new_df], axis=0)

    X.to_csv('./resources/preprocessed_data/data.csv', encoding='utf-8')
    Y.to_csv('./resources/preprocessed_data/target.csv', encoding='utf-8')

if __name__ == '__main__':
    main()
