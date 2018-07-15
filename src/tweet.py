import tweepy
import datetime
from urllib import request
from bs4 import BeautifulSoup
from sklearn.externals.joblib import load
from keras.models import load_model
import numpy as np
import re
import unicodedata

from preprocess import extract_time, extract_weather
import keys
CK = keys.CONSUMER_KEY
CS = keys.CONSUMER_SECRET
AT = keys.ACCESS_TOKEN
AS = keys.ACCESS_TOKEN_SECRET

auth = tweepy.OAuthHandler(CK, CS)
auth.set_access_token(AT, AS)
api = tweepy.API(auth)


clfs = ['svm', 'rf', 'xgb', 'krs']
clf_names = {'svm': 'SVM', 'rf': 'RandomForest', 'xgb': 'XGBoost', 'krs': 'Deep Learning'}

class Listener(tweepy.StreamListener):

    def on_status(self, status):
        status.created_at += datetime.timedelta(hours=9)

        # リプライが来たら返信
        if str(status.in_reply_to_screen_name) == "test_kwmt":
            text = unicodedata.normalize('NFKC', status.text)
            minute = re.compile('([0-9]+)分').search(text)
            now = datetime.datetime.now()
            if minute:
                now += datetime.timedelta(minutes = int(minute.group(1)))
            time_list = extract_time(now.year, now.month, now.day, now.hour, now.minute, now.weekday()+1, int(now.strftime('%j')))
            weather_list = yahoo_weather()
            X = np.array(time_list + weather_list).reshape(1,-1)
            text = ''
            for clf in clfs:
                if clf == 'krs':
                    Y_pred = load_model('estimators/' + clf + '.h5').predict(X)[0]
                else:
                    Y_pred = load('estimators/' + clf + '.pkl').predict(X)
                text += clf_names[clf] + '・・・ {:.2f}人\n'.format(Y_pred[0])

            tweet = '@{0}\nラーメン二郎仙台店の行列\n{1}時{2}分の予測\n{3}'.format(str(status.user.screen_name), now.hour, now.minute, text)
            api.update_status(status=tweet, in_reply_to_status_id=status.id)
            print(X)
            print(tweet)

        return True

    def on_error(self, status_code):
        print('Got an error with status code: ' + str(status_code))
        return True

    def on_timeout(self):
        print('Timeout...')
        return True

def yahoo_weather():
    weather = ['晴', '曇', '雨', '雪', '雷', '風']
    with request.urlopen('https://weather.yahoo.co.jp/weather/jp/4/3410.html') as response:
        soup = BeautifulSoup(response.read(), 'html.parser')
        p = soup.find_all('p')
        weather_text = ''
        for elem in p:
            try:
                if elem.get('class').pop(0) in 'pict':
                    text = str(elem)
                    break
            except:
                pass
        em = soup.find_all('em')
        temp_h = float(em[0].get_text())
        temp_l = float(em[1].get_text())
    weather_list = [temp_h, temp_l] + [1 if t in text else 0 for t in weather]
    return weather_list


def main():
    auth = tweepy.OAuthHandler(CK, CS)
    auth.set_access_token(AT, AS)
    listener = Listener()
    stream = tweepy.Stream(auth, listener)
    stream.userstream()

if __name__ == '__main__':
    while True:
        try:
            main()
            break
        except AttributeError:
            print('AttributeError')
