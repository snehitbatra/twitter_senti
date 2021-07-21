import streamlit as st
import warnings
warnings.filterwarnings("ignore")
# EDA Pkgs
import pandas as pd
import numpy as np
import pandas as pd
import tweepy
import json
from tweepy import OAuthHandler
import re
import textblob
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import openpyxl
import time
import tqdm
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

def main():
    """ Common ML Dataset Explorer """
    #st.title("Live twitter Sentiment analysis")
    #st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

    html_temp = """
	<div style="background-color:tomato;"><p style="color:white;font-size:40px;padding:9px">Live twitter Sentiment analysis</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Select a topic which you'd like to get the sentiment analysis on :")

    ################# Twitter API Connection #######################
    consumer_key = "fGkPET5RRpV609RdrB3LItZRS"
    consumer_secret = "Tk4W90y92fVGZoyjS5gwhlFLETj8fGR09zY3EhbeY5FST9eQXm"
    access_token = "1391611633923878913-s4oQhAzn93xQxhyEGxTEjG0Xa7yBpO"
    access_secret = "txdxHtsMr1brVGWWhDPfdFZeX1t8y5EnYFa4uNa6sluN0"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    def get_tweets(search_term,Count):
        no_of_tweets = 100
        twitter = tweepy.Cursor(api.search, q=search_term).items(no_of_tweets)
        from textblob import TextBlob
        X = []
        for tweet in twitter:
            X.append(tweet.text)
        return X
    def clean_text(review):
        import nltk
        import re
        from nltk.corpus import stopwords
        stopword = stopwords.words('english')
        review = re.sub('[^a-zA-Z]', ' ', str(review))
        review = review.split()

        ap = []
        for word in review:
            if word not in stopwords.words('english'):
                ap.append(word)
        review = ' '.join(ap)
        return review
    def prepCloud(Topic_text,Topic):
        Topic = str(Topic).lower()
        Topic=' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
        Topic = re.split("\s+",str(Topic))
        stopwords = set(STOPWORDS)
        stopwords.update(Topic) ### Add our topic in Stopwords, so it doesnt appear in wordClous
        ###
        text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
        return text_new
    import pandas as pd
    Topic = str()
    Topic = str(st.text_input("Enter the topic you are interested in (Press Enter once done)"))
    if len(Topic) > 0 :
        with st.spinner("Please wait, Tweets are being extracted"):
                X=get_tweets(Topic , Count=200)
        data = pd.DataFrame(X)
        data.columns=['tweets']
        Xx = data['tweets'].apply(clean_text)
        data['tweets'] = Xx
        import joblib
        loaded_model = joblib.load('filename.pkl')
        y_pred = loaded_model.predict(data['tweets'])
        data['sentiment'] = y_pred
        from matplotlib import pyplot as plt
        twee = ['positive', 'negative']
        for i in range(len(data)):
            if data['sentiment'][i]==4:
                data['sentiment'][i]='positive'
            if data['sentiment'][i]==0:
                data['sentiment'][i]='negative'
        data['tweets'] = Xx
        #st.write("Total Positive Tweets are : {}".format(positive[4]))
        st.write("Total Positive Tweets are : {}".format(len(data[data["sentiment"]=="positive"])))
        st.write("Total Negative Tweets are : {}".format(len(data[data["sentiment"]=='negative'])))
        if st.button("See the Extracted Data"):
            st.success("Below is the Extracted Data :")
            st.write(data.head(50))
        if st.button("See the count plot of the tweet sentiments"):
            st.success("Generating A Count Plot")
            st.subheader(" Count Plot for Different Sentiments")
            st.write(sns.countplot(data["sentiment"]))
            st.pyplot()
        if st.button("See the word cloud for all positive things said about ".format(Topic)):
            import nltk
            import re
            from nltk.corpus import stopwords
            stopword=stopwords.words('english')
            new_stopwords = ["@", "RT"]
            stopword.extend(new_stopwords)
            subset=data[data.sentiment=='positive']
            text=subset.tweets.values
            wc= WordCloud(background_color="black",max_words=4000,stopwords=stopword)
            wc.generate(" ".join(text))
            st.write(plt.title("Words frequented in Positive Comments", fontsize=20))
            st.write(plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98))
            st.pyplot()
        if st.button("See the word cloud for negative things said about ".format(Topic)):
            import nltk
            import re
            from nltk.corpus import stopwords
            stopword=stopwords.words('english')
            new_stopwords = ["@", "RT","HTTPS","https"]
            stopword.extend(new_stopwords)
            subset=data[data.sentiment=='negative']
            text=subset.tweets.values
            wc= WordCloud(background_color="black",max_words=4000,stopwords=stopword)
            wc.generate(" ".join(text))
            st.write(plt.title("Words frequented in Negative Comments", fontsize=20))
            st.write(plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98))
            st.pyplot()





if __name__ == "__main__":
    main()


        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.

    #st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Neutral"])))
