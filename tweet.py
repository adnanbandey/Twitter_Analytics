import streamlit as st
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
from tweepy import OAuthHandler
import json

import seaborn as sns

# from textblob.sentiments import NaiveBayesAnalyzers
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import textmining

import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
from matplotlib.pyplot import yticks
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

# from nltk.corpus import stopwords
# import string
# import nltk
# import textmining
import matplotlib.pyplot as plt
# from wordcloud import WordCloud,STOPWORDS

from pandas.io.json import json_normalize

consumer_key = 'xx'
consumer_secret = 'yy'
access_token = 'aa'
access_token_secret = 'bb'

from tweepy import API


class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client


class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        return auth
twitter_client = TwitterClient()
api = twitter_client.get_twitter_client_api()
maxtweetsperquery = 100

def app():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Twitter Analytics 👾</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # st.title("Twitter Analytics 👾")

    activities = ["Tweet Analytics", "Generate Twitter Data"]
    choice = st.sidebar.selectbox("Select your task : ", activities)
    if choice == "Tweet Analytics":

        st.subheader("Fetches tweets for a search term and analyses it")
        st.subheader("This app performs the following tasks :")
        st.write("1. Fetches the most recent tweets from famous or verified people")
        st.write("2. Generates a Word Cloud about the tweets and description of the people tweeting about it")
        st.write("3. Performs Sentiment Analysis and displays it in a form of visualization")
        st.write("4. Display top hashtags being used with count")
        st.write("5. Users related analytics")

        raw_text = st.text_area("Enter the exact search term you want tweets for",key='one')
        text_num = st.text_area("Enter the num of tweets you want",key='two')
        st.markdown(" Check other functionalities from the sidebar at the left")
        Analyzer_choice = st.selectbox("Select the Activities",
                                       ["Show Recent Tweets from Verified or Famous people 👀", "Generate WordCloud ☁",
                                        "Visualize the Sentiment Analysis 📊","Top Hashtags trending about the term 🤳🏽",
                                        "Analytics related to users 🔧"])

        if st.button("Analyze"):

            if Analyzer_choice == "Show Recent Tweets from Verified or Famous people 👀":

                st.success("Fetching last n Tweets")

                def Show_Recent_Tweets(raw_text, text_num):

                    tweets = api.search(q=raw_text, count=maxtweetsperquery, lang='en', tweet_mode='extended')
                    norm_df = pd.DataFrame()
                    for tweet in tweets:
                        temp = pd.json_normalize(tweet._json)  # put json data in brackets
                        norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                    lowest_id = np.min(norm_df['id'])
                    lenght = len(tweets)
                    de = int(text_num)
                    while lenght < de:
                        tweets = api.search(q=raw_text, result_type='recent', count=maxtweetsperquery, lang='en',
                                            tweet_mode='extended')
                        for tweet in tweets:
                            temp = pd.json_normalize(tweet._json)  # put json data in brackets
                            norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                        lowest_id = np.min(norm_df['id'])
                        lenght += len(tweets)

                    norm_df['main_text'] = ""
                    norm_df['Tweet Status'] = 'Normal Tweet'
                    norm_df['Tweet Status'][norm_df['full_text'].str.contains('RT')] = 'Retweet'

                    for i in range(norm_df.shape[0]):
                        if norm_df['Tweet Status'].iloc[i] == 'Retweet':
                            norm_df['main_text'].iloc[i] = norm_df['retweeted_status.full_text'].iloc[i]
                        else:
                            norm_df['main_text'].iloc[i] = norm_df['full_text'].iloc[i]

                    norm_df1=norm_df[(norm_df['user.verified'] == True) | (norm_df['user.followers_count'] > 50000)]
                    # norm_df1=norm_df1.drop_duplicated()

                    return norm_df1[['main_text','user.name','user.description','user.followers_count']].style.highlight_max(axis=0)

                recent_tweets=Show_Recent_Tweets(raw_text,text_num)

                st.table(recent_tweets)



            elif Analyzer_choice == "Generate WordCloud ☁":

                st.success("Generating Word Cloud")

                def cleanTxt(text):
                    text = re.sub('@[A-Za-z0–9]+', '', str(text))  # Removing @mentions
                    text = re.sub('#', '', text)  # Removing '#' hash tag
                    text = re.sub('RT[\s]+', '', text)  # Removing RT
                    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
                    return text

                def gen_wordcloud():
                    norm_df = pd.DataFrame()
                    tweets = api.search(q=raw_text, count=maxtweetsperquery, lang='en', tweet_mode='extended')
                    for tweet in tweets:
                        temp = pd.json_normalize(tweet._json)  # put json data in brackets
                        norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                    lowest_id = np.min(norm_df['id'])
                    lenght = len(tweets)
                    de = int(text_num)
                    while lenght < de:
                        tweets = api.search(q=raw_text, result_type='recent', count=maxtweetsperquery, lang='en',
                                            tweet_mode='extended')
                        for tweet in tweets:
                            temp = pd.json_normalize(tweet._json)  # put json data in brackets
                            norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                        lowest_id = np.min(norm_df['id'])
                        lenght += len(tweets)

                    norm_df['main_text'] = ""
                    norm_df['Tweet Status'] = 'Normal Tweet'
                    norm_df['Tweet Status'][norm_df['full_text'].str.contains('RT')] = 'Retweet'
                    for i in range(norm_df.shape[0]):
                        if norm_df['Tweet Status'].iloc[i] == 'Retweet':
                            norm_df['main_text'].iloc[i] = norm_df['retweeted_status.full_text'].iloc[i]
                        else:
                            norm_df['main_text'].iloc[i] = norm_df['full_text'].iloc[i]

                    norm_df['main_text_clean'] = norm_df['main_text'].apply(cleanTxt)

                    allWords = ' '.join([twts for twts in norm_df['main_text_clean']])
                    wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
                    # plt.figure(figsize=(10,10))
                    plt.imshow(wordCloud, interpolation="bilinear")
                    plt.axis('off')
                    plt.show()
                    plt.savefig('AB.jpg')
                    img = Image.open("AB.jpg")
                    return img

                def gen_wordcloud1():
                    norm_df = pd.DataFrame()
                    tweets = api.search(q=raw_text, count=maxtweetsperquery, lang='en', tweet_mode='extended')
                    for tweet in tweets:
                        temp = pd.json_normalize(tweet._json)  # put json data in brackets
                        norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                    lowest_id = np.min(norm_df['id'])
                    lenght = len(tweets)
                    de = int(text_num)
                    while lenght < de:
                        tweets = api.search(q=raw_text, result_type='recent', count=maxtweetsperquery, lang='en',
                                            tweet_mode='extended')
                        for tweet in tweets:
                            temp = pd.json_normalize(tweet._json)  # put json data in brackets
                            norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                        lowest_id = np.min(norm_df['id'])
                        lenght += len(tweets)

                    norm_df['main_text'] = ""
                    norm_df['Tweet Status'] = 'Normal Tweet'
                    norm_df['Tweet Status'][norm_df['full_text'].str.contains('RT')] = 'Retweet'
                    for i in range(norm_df.shape[0]):
                        if norm_df['Tweet Status'].iloc[i] == 'Retweet':
                            norm_df['main_text'].iloc[i] = norm_df['retweeted_status.full_text'].iloc[i]
                        else:
                            norm_df['main_text'].iloc[i] = norm_df['full_text'].iloc[i]

                    norm_df['main_text_clean'] = norm_df['main_text'].apply(cleanTxt)

                    allWords1 = ' '.join([twts for twts in norm_df['user.description']])
                    wordCloud1 = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords1)
                    # plt.figure(figsize=(10,10))
                    plt.imshow(wordCloud1, interpolation="bilinear")
                    plt.axis('off')
                    plt.show()
                    plt.savefig('AB.jpg')
                    img1 = Image.open("AB.jpg")
                    return img1

                img = gen_wordcloud()

                st.image(img,caption="WordCloud of the tweets",width=750)

                img1 = gen_wordcloud1()

                st.image(img1, caption="WordCloud of the Description of Users", width=750)

            elif Analyzer_choice == "Visualize the Sentiment Analysis 📊":

                def Plot_Analysis():

                    st.success("Generating Visualisation for Sentiment Analysis 📉")

                    def cleanTxt(text):
                        text = re.sub('@[A-Za-z0–9]+', '', str(text))  # Removing @mentions
                        text = re.sub('#', '', text)  # Removing '#' hash tag
                        text = re.sub('RT[\s]+', '', text)  # Removing RT
                        text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
                        return text

                    def getSubjectivity(text):
                        return TextBlob(text).sentiment.subjectivity

                    # Create a function to get the polarity
                    def getPolarity(text):
                        return TextBlob(text).sentiment.polarity

                    def getAnalysis(score):
                        if score < 0:
                            return 'Negative'
                        elif score == 0:
                            return 'Neutral'
                        else:
                            return 'Positive'

                    norm_df = pd.DataFrame()
                    tweets = api.search(q='covid', count=maxtweetsperquery, lang='en', tweet_mode='extended')
                    norm_df = pd.DataFrame()

                    for tweet in tweets:
                        temp = pd.json_normalize(tweet._json)  # put json data in brackets
                        norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                        lowest_id = np.min(norm_df['id'])
                        lenght = len(tweets)
                        de = int(200)
                    while lenght < de:
                        tweets = api.search(q=str, result_type='recent', count=maxtweetsperquery, lang='en',
                                            tweet_mode='extended')
                        for tweet in tweets:
                            temp = pd.json_normalize(tweet._json)  # put json data in brackets
                            norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                        lowest_id = np.min(norm_df['id'])
                        lenght += len(tweets)

                    norm_df['main_text'] = ""
                    norm_df['Tweet Status'] = ""
                    norm_df['Tweet Status'] = 'Normal Tweet'
                    norm_df['Tweet Status'][norm_df['full_text'].str.contains('RT')] = 'Retweet'

                    for i in range(norm_df.shape[0]):
                        if norm_df['Tweet Status'].iloc[i] == 'Retweet':
                            norm_df['main_text'].iloc[i] = norm_df['retweeted_status.full_text'].iloc[i]
                        else:
                            norm_df['main_text'].iloc[i] = norm_df['full_text'].iloc[i]

                    norm_df1_new = norm_df
                    # Clean the tweets
                    norm_df1_new['main_text_clean'] = norm_df1_new['main_text'].apply(cleanTxt)

                    # # Create two new columns 'Subjectivity' & 'Polarity'
                    norm_df1_new['Subjectivity'] = norm_df1_new['main_text_clean'].apply(getSubjectivity)
                    norm_df1_new['Polarity'] = norm_df1_new['main_text_clean'].apply(getPolarity)

                    norm_df1_new['Analysis'] = norm_df1_new['Polarity'].apply(getAnalysis)

                    return norm_df1_new

                df=Plot_Analysis()

                st.write(sns.countplot(x=df["Analysis"], data=df))

                st.pyplot(use_container_width=True)

            elif Analyzer_choice == "Top Hashtags trending about the term 🤳🏽":

                def Hashtags():

                    st.success("Top Hashtags with count 🤳🏽")

                    norm_df = pd.DataFrame()
                    tweets = api.search(q='covid', count=maxtweetsperquery, lang='en', tweet_mode='extended')
                    norm_df = pd.DataFrame()

                    for tweet in tweets:
                        temp = pd.json_normalize(tweet._json)  # put json data in brackets
                        norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                        lowest_id = np.min(norm_df['id'])
                        lenght = len(tweets)
                        de = int(200)
                    while lenght < de:
                        tweets = api.search(q=str, result_type='recent', count=maxtweetsperquery, lang='en',
                                            tweet_mode='extended')
                        for tweet in tweets:
                            temp = pd.json_normalize(tweet._json)  # put json data in brackets
                            norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                        lowest_id = np.min(norm_df['id'])
                        lenght += len(tweets)

                    list1 = []
                    for i in range(norm_df.shape[0]):
                        for j in range(len(norm_df['entities.hashtags'].iloc[i])):
                            pp = norm_df['entities.hashtags'].iloc[i][j]['text']
                            list1.append(pp)
                    df = pd.DataFrame(list1, columns=['Hashtags'])
                    top5 = df['Hashtags'].value_counts().head(10)
                    top5 = top5.to_frame().reset_index()
                    top5.columns = ['Hashtags', 'Count']

                    return top5

                df=Hashtags()

                st.table(df)

            else:
                st.success("Analysing user related data")

                def cleanTxt(text):
                    text = re.sub('@[A-Za-z0–9]+', '', str(text))  # Removing @mentions
                    text = re.sub('#', '', text)  # Removing '#' hash tag
                    text = re.sub('RT[\s]+', '', text)  # Removing RT
                    text = re.sub('https?:\/\/\S+', '', text)  # Removing hyperlink
                    return text

                def user_analytics():
                    norm_df = pd.DataFrame()
                    tweets = api.search(q=raw_text, count=maxtweetsperquery, lang='en', tweet_mode='extended')
                    for tweet in tweets:
                        temp = pd.json_normalize(tweet._json)  # put json data in brackets
                        norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                    lowest_id = np.min(norm_df['id'])
                    lenght = len(tweets)
                    de = int(text_num)
                    while lenght < de:
                        tweets = api.search(q=raw_text, result_type='recent', count=maxtweetsperquery, lang='en',
                                            tweet_mode='extended')
                        for tweet in tweets:
                            temp = pd.json_normalize(tweet._json)  # put json data in brackets
                            norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                        lowest_id = np.min(norm_df['id'])
                        lenght += len(tweets)

                    norm_df['main_text'] = ""
                    norm_df['Tweet Status'] = 'Normal Tweet'
                    norm_df['Tweet Status'][norm_df['full_text'].str.contains('RT')] = 'Retweet'
                    for i in range(norm_df.shape[0]):
                        if norm_df['Tweet Status'].iloc[i] == 'Retweet':
                            norm_df['main_text'].iloc[i] = norm_df['retweeted_status.full_text'].iloc[i]
                        else:
                            norm_df['main_text'].iloc[i] = norm_df['full_text'].iloc[i]

                    norm_df['main_text_clean'] = norm_df['main_text'].apply(cleanTxt)

                    norm_df = norm_df.dropna(thresh=norm_df.shape[0] * 0.1, how='all', axis=1)
                    norm_df['Device'] = 'Others'
                    norm_df['Device'][norm_df['source'].str.contains('iphone')] = 'Iphone'
                    norm_df['Device'][norm_df['source'].str.contains('android')] = 'Andriod'

                    q = norm_df['user.followers_count'].quantile(0.85)
                    norm_df_clean = norm_df[norm_df['user.followers_count'] < q]
                    return norm_df_clean
                    # plt.figure(figsize=(10, 7))
                    # plt.title(label='Users followers distribution wrt Device', size=17)


                df = user_analytics()

                #users mentioned
                st.subheader("Top users mentioned")
                list2 = []
                for i in range(df.shape[0]):
                    for j in range(len(df['entities.user_mentions'].iloc[i])):
                        qq = df['entities.user_mentions'].iloc[i][j]['name']
                        list2.append(qq)
                df1 = pd.DataFrame(list2, columns=['Users'])
                top5_df1 = df1['Users'].value_counts().head(10)
                top5_df1 = top5_df1.to_frame().reset_index()
                top5_df1.columns = ['Users mentioned', 'Count']
                st.table(top5_df1)

                st.subheader("Users distribution with no picture and low followers")

                #user distribution with no pp picture and low followers
                df['user.default_profile_image'].replace(False, 'Profile picture not present', inplace=True)
                df['user.default_profile_image'].replace(True, 'Profile Picture present', inplace=True)
                df_11 = df['user.default_profile_image'].value_counts()
                df_11_pp = df_11.to_frame().reset_index()
                df_11_pp.columns = ['Users', 'Count']
                st.table(df_11_pp)

                #year users created accounts
                st.subheader("Different years these users created their accounts in")
                df['Year'] = np.nan
                df['Year'] = df['user.created_at'].apply(lambda x: x[-4:])
                top_10 = df['Year'].value_counts().head(10)
                top_10_df = top_10.to_frame().reset_index()
                top_10_df.columns = ['Year', 'Count']
                top_10_df = top_10_df.sort_values(by=['Year'], ascending=False)
                st.write(sns.barplot(x=top_10_df["Year"], y=top_10_df["Count"], data=top_10_df))
                # plt.title(label='Different years users created their accounts', size=17)
                plt.ylabel('Count', size=10)
                plt.xlabel('Year', size=10)
                # xticks(rotation=10, size=10)
                st.pyplot(use_container_width=True)

                #users followers count
                st.subheader("Users followers distribution wrt Device")
                st.write(sns.boxplot(x="Device", y="user.followers_count", data=df))
                # plt.title(label='Users followers distribution wrt Device', size=17)
                plt.ylabel('Followers count', size=10)
                plt.xlabel('Device', size=10)
                xticks(rotation=40, size=10)
                st.pyplot(use_container_width=True)

    else:

        st.subheader("This tool fetches the last N tweets for the twitter search term")
        # st.write("5. Analyzes Sentiments of tweets and adds an additional column for it")

        raw_text = st.text_area("*Enter the term you want tweets you want to see*")
        text_num = st.text_area("*Enter the no of tweets you want to see*")

        def get_data():

            norm_df = pd.DataFrame()
            tweets = api.search(q=raw_text,count=maxtweetsperquery, lang='en', tweet_mode='extended')
            norm_df = pd.DataFrame()

            for tweet in tweets:
                temp = pd.json_normalize(tweet._json)  # put json data in brackets
                norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
            lowest_id = np.min(norm_df['id'])
            lenght = len(tweets)
            de = int(text_num)
            while lenght < de:
                tweets = api.search(q=raw_text, result_type='recent', count=maxtweetsperquery, lang='en',
                                    tweet_mode='extended')
                for tweet in tweets:
                    temp = pd.json_normalize(tweet._json)  # put json data in brackets
                    norm_df = pd.concat([norm_df, temp], ignore_index=True, axis=0)
                lowest_id = np.min(norm_df['id'])
                lenght += len(tweets)

            norm_df['main_text'] = ""
            norm_df['Tweet Status'] = ""
            norm_df['Tweet Status'] = 'Normal Tweet'
            norm_df['Tweet Status'][norm_df['full_text'].str.contains('RT')] = 'Retweet'

            for i in range(norm_df.shape[0]):
                if norm_df['Tweet Status'].iloc[i] == 'Retweet':
                    norm_df['main_text'].iloc[i] = norm_df['retweeted_status.full_text'].iloc[i]
                else:
                    norm_df['main_text'].iloc[i] = norm_df['full_text'].iloc[i]

            norm_df1_new = norm_df

            return norm_df1_new['main_text']

        if st.button("Show Data"):

            st.success("Fetching Last N Tweets")

            df = get_data()

            st.table(df)

            st.subheader('Twitter Analyzer App')

    st.subheader('                                                                                        Created By : Adnan Bandey ')

# port=int(os.getenv("PORT"))
if __name__ == "__main__":
    # app.run(host='0.0.0.0',port=port)
    app()
