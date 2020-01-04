import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as pkt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
import pickle

df = pd.read_csv("Datasets/emails.csv/emails.csv")


df.drop_duplicates(inplace=True)

df['text'] = df['text'].map(lambda text: re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())

df['text'] = df['text'].map(lambda text:text[1:])

ps = PorterStemmer()

word_list = df['text'].tolist()

stopwords = set(stopwords.words('english'))


new_words_list = []
for i in word_list:
    new_words_list.append([x for x in i if x not in stopwords])


stem_new_words = []
for i in new_words_list:
    stem_new_words.append([ps.stem(x) for x in i])


joined_words = []
for i in stem_new_words:
    
    joined_words.append(' '.join(i))


cv = CountVectorizer()


X = cv.fit_transform(joined_words)


y = df.iloc[:,1].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
classifier.fit(X_train,y_train)




y_pred = classifier.predict(X_test)


with open("nbspam.pkl","wb") as f:
	pickle.dump(classifier,f)


