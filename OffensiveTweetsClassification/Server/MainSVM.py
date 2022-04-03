import pandas as pd
import numpy as np
import regex as re
import nltk
from nltk.corpus import stopwords
from camel_tools.utils.dediac import dediac_ar
from nltk.stem.isri import ISRIStemmer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

dataSet = pd.read_csv("data.csv", dtype={0: 'str', 1: 'float'}, encoding='utf-8')
inputData = dataSet.iloc[:, 0].values
outputClass = dataSet.iloc[:, 1].values
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit([outputClass])
outputClass = imp.transform([outputClass])

number_of_rows = len(inputData)
stopWords = nltk.download("stopwords")
stop_words = stopwords.words('arabic')
cleanedTweets = []

for idx, word in enumerate(stop_words):
    stop_words[idx] = dediac_ar(word)

for i in range(0, number_of_rows):
    tweet = inputData[i]
    tweet = re.sub("[^\u0600-\u06FF ]", '', tweet)  # Remove all special characters and numbers.
    tweet = re.sub(r'(.)\1+', r'\1', tweet)
    tweetWords = tweet.split()  # Split the words in the review.
    st = ISRIStemmer()  # Creating a stemmer object.
    cleanedTweet = [(dediac_ar(word)) for word in tweetWords if word not in set(stop_words)]
    cleanedTweet = u' '.join(cleanedTweet)  # Join all words together after stemming them and removing stopwords.
    cleanedTweets.append(cleanedTweet)

vectorizer = CountVectorizer()
data = vectorizer.fit_transform(cleanedTweets).toarray()
tweet_train, tweet_test, offensive_train, offensive_test = train_test_split(data, dataSet.iloc[:, -1].values,
                                                                            test_size=0.2, random_state=0)
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(tweet_train, offensive_train)

pickle.dump(classifier, open("SVM_BOW_without_stemming.sav", 'wb'))

offensive_prediction = classifier.predict(tweet_test)

cm = confusion_matrix(offensive_test, offensive_prediction)

print("The Confusion Matrix: \n", cm)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
pr = tp / (tp + fp)
rec = tp / (tp + fn)
print("The Precision Is : ", pr)
print("The Recall Is : ", rec)
print("The F1 - Score Is : ", (2 * pr * rec) / (pr + rec))
print("Accuracy Is : ", accuracy_score(offensive_test, offensive_prediction))
