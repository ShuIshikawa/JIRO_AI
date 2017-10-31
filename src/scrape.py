
from bs4 import BeautifulSoup
import pandas as pd
import time
import calendar

def main():
    with open('./resources/webpage.html', 'r') as fp:
        soup = BeautifulSoup(fp)


        clms = ['text', 'year', 'month', 'day', 'hour', 'minute', 'second', 'week_name', 'week_num', 'num_of_day', 'num_of_week']
        tweets_df = pd.DataFrame(columns=clms)

        i = 0

        div = soup.find_all('div')
        for elem in div:
            try:
                if elem.get('class').pop(0) in 'content':

                    tw_time = ''
                    tw_text = ''

                    for span in elem.find_all('span'):
                        try:
                            if span.get('class').pop(0) in '_timestamp js-short-timestamp ':
                                tw_time = str(span['data-time'])
                                break
                        except:
                            pass

                    for p in elem.find_all('p'):
                        try:
                            if p.get('class').pop(0) in "TweetTextSize js-tweet-text tweet-text":
                                tw_text = str(p.contents).replace('<a class="twitter-atreply pretty-link js-nav" data-mentioned-user-id="43241089" dir="ltr" href="https://twitter.com/jirolian">', '')
                                tw_text = tw_text.replace('</a>', '')
                                tw_text = tw_text.replace('<s>', '')
                                tw_text = tw_text.replace('</s>', '')
                                tw_text = tw_text.replace('<b>', '')
                                tw_text = tw_text.replace('</b>', '')
                                tw_text = tw_text.replace('<strong>', '')
                                tw_text = tw_text.replace('</strong>', '')
                                tw_text = tw_text.replace('[', '')
                                tw_text = tw_text.replace(']', '')
                                tw_text = tw_text.replace(',', '')
                                tw_text = tw_text.replace('\'', '')
                                tw_text = tw_text.replace('\n', ' ')
                                tw_text = tw_text.replace('\\n', ' ')
                                tw_text = tw_text.replace('\r', ' ')
                                tw_text = tw_text.replace('\\r', ' ')
                                tw_text = tw_text.replace('\r\n', ' ')
                                tw_text = tw_text.replace('\\r\\n', ' ')
                                tw_text = tw_text.replace('@jirolian', ' ')
                                break
                        except:
                            pass

                    if not(tw_time == '' or tw_text == ''):
                        time_data = time.localtime(int(tw_time))
                        if 9 < int(time.strftime("%H", time_data)) < 23 and int(time.strftime("%Y", time_data)) > 2011:
                            new_df = pd.DataFrame([[tw_text,
                                                    int(time.strftime("%Y", time_data)),
                                                    int(time.strftime("%m", time_data)),
                                                    int(time.strftime("%d", time_data)),
                                                    int(time.strftime("%H", time_data)),
                                                    int(time.strftime("%M", time_data)),
                                                    int(time.strftime("%S", time_data)),
                                                    str(time.strftime("%a", time_data)),
                                                    int(time.strftime("%w", time_data)),
                                                    int(time.strftime("%j", time_data)),
                                                    int(time.strftime("%U", time_data))]])
                            new_df.columns = clms
                            new_df.index = [i]
                            tweets_df = pd.concat([tweets_df, new_df], axis=0)
                            i = i + 1
            except:
                pass

    print(tweets_df)
    tweets_df.to_csv('./resources/tweet_data.csv', encoding='utf-8')
    print('\n{0} tweets saved'.format(i))

if __name__ == '__main__':
    main()
