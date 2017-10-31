
import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split

import holiday_list
holidays = holiday_list.holiday


def extract_target(text):
    if ('待ちなし' in text) or ('待ち無し' in text):
        return 0
    else:
        text = unicodedata.normalize('NFKC', text)
        s1 = re.compile('([0-9]+)人').search(text)
        s2 = re.compile('待ち([0-9]+)').search(text)
        s3 = re.compile('並び([0-9]+)').search(text)
        if s1:
            gro = int(s1.group(1))
        elif s2:
            gro = int(s2.group(1))
        elif s3:
            gro = int(s3.group(1))
        else:
            return 100

        if gro < 60:
            return gro
        else:
            return 100

def extract_time(year, month, day, hour, minute, week_num, num_of_day, num_of_week):

    if '{}/{}/{}'.format(year, month, day) in holidays:
        hol = 1
    else:
        hol = 0

    time_list = [1 if month == 1 else 0,
                 1 if month == 2 else 0,
                 1 if month == 3 else 0,
                 1 if month == 4 else 0,
                 1 if month == 5 else 0,
                 1 if month == 6 else 0,
                 1 if month == 7 else 0,
                 1 if month == 8 else 0,
                 1 if month == 9 else 0,
                 1 if month == 10 else 0,
                 1 if month == 11 else 0,
                 1 if month == 12 else 0,
                 day,
                 hour,
                 minute,
                 1 if week_num % 7 == 0 else 0,
                 1 if week_num == 1 else 0,
                 1 if week_num == 2 else 0,
                 1 if week_num == 3 else 0,
                 1 if week_num == 4 else 0,
                 1 if week_num == 5 else 0,
                 1 if week_num == 6 else 0,
                 hol,
                 num_of_day,
                 num_of_week]
    return time_list

def extract_weather(year, month, day):
    weather_df = pd.read_csv('./resources/weather_data.csv')
    for key, weather in weather_df.iterrows():
        if weather['day'] == '{}/{}/{}'.format(year, month, day):
            weather_list = [float(weather['temp_h']),
                            float(weather['temp_l']),
                            1 if '晴' in weather['weat_d'] else 0,
                            1 if '曇' in weather['weat_d'] else 0,
                            1 if '雨' in weather['weat_d'] else 0,
                            1 if '雪' in weather['weat_d'] else 0,
                            1 if '雷' in weather['weat_d'] else 0,
                            1 if '風' in weather['weat_d'] else 0]
            break
    return weather_list

def main():
    tweet_df = pd.read_csv('./resources/tweet_data.csv')

    clms = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
            'day', 'hour', 'minute',
            'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'hol',
            'num_of_day', 'num_of_week',
            'temp_h', 'temp_l',
            'weat_sun', 'weat_cloud', 'weat_rain', 'weat_snow',
            'weat_light', 'weat_storm']

    X = pd.DataFrame(columns=clms)
    Y = pd.DataFrame(columns=['target'])

    j = 0
    for key, tweet in tweet_df.iterrows():
        target = extract_target(tweet['text'])
        if target < 100:

            Y = pd.concat([Y, pd.DataFrame({'target': [target]})], axis=0)
            new_df = pd.DataFrame([extract_time(int(tweet['year']),
                                                int(tweet['month']),
                                                int(tweet['day']),
                                                int(tweet['hour']),
                                                int(tweet['minute']),
                                                int(tweet['week_num']),
                                                int(tweet['num_of_day']),
                                                int(tweet['num_of_week']))
                                   + extract_weather(str(tweet['year']),
                                                     str(tweet['month']),
                                                     str(tweet['day']))],
                                  columns=clms)
            X = pd.concat([X, new_df], axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=1)

    X_train.to_pickle('./resources/train/data.pkl')
    Y_train.to_pickle('./resources/train/target.pkl')
    X_test.to_pickle('./resources/test/data.pkl')
    Y_test.to_pickle('./resources/test/target.pkl')

if __name__ == '__main__':
    main()
