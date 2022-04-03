import json
import pickle
import numpy as np
import pandas as pd
import regex as re
from camel_tools.utils.dediac import dediac_ar
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS, cross_origin


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_confusion_matrix(y_test, spam_prediction):
    cm = confusion_matrix(y_test, spam_prediction)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    pr = tp / (tp + fp)
    rec = tp / (tp + fn)
    result = dict()
    result["tp"] = tp
    result["fp"] = fp
    result["fn"] = fn
    result["tn"] = tn
    result["Precision"] = round(pr * 100, 3)
    result["Recall"] = round(rec * 100, 3)
    result["F1-score"] = round((2 * pr * rec) / (pr + rec) * 100, 3)
    result["Accuracy"] = round(accuracy_score(y_test, spam_prediction) * 100, 3)
    return result


def load_model(file_name, run_type, tweet_test, offensive_test, data=""):
    if file_name.startswith("ANN"):
        reconstructed_model = keras.models.load_model(file_name)
        offensive_prediction = reconstructed_model.predict(tweet_test if run_type == 0 else data) > 0.5
        if run_type == 0:
            result = get_confusion_matrix(offensive_test, offensive_prediction)
            return result
        else:
            if offensive_prediction:
                result2 = dict()
                result2["class"] = "Offensive"
                return result2
            else:
                result2 = dict()
                result2["class"] = "not Offensive"
                return result2

    loaded_model = pickle.load(open(file_name, 'rb'))
    offensive_prediction = loaded_model.predict(tweet_test if run_type == 0 else data)
    if run_type == 0:
        result = get_confusion_matrix(offensive_test, offensive_prediction)
        return result
    else:
        if offensive_prediction == 1.0:
            result2 = dict()
            result2["class"] = "Offensive"
            return result2
        else:
            result2 = dict()
            result2["class"] = "not Offensive"
            return result2


app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'NLP'
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/": {"origins": "http://localhost:3000"}})
api = Api(app)


@app.route("/", methods=['POST'])
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def home():
    var = dict(request.get_json())
    models = [
        "Decision_Tree",
        "Random_Forest",
        "NB",
        "SVM",
        "KNN",
        "ANN"
    ]
    dataSet = pd.read_csv("data.csv", dtype={0: 'str', 1: 'float'}, encoding='utf-8')
    inputData = dataSet.iloc[:, 0].values
    number_of_rows = len(inputData)
    stop_words = stopwords.words('arabic')

    for idx, word in enumerate(stop_words):
        stop_words[idx] = dediac_ar(word)

    choice = str(var.get("model"))
    inputType = str(var.get("mode"))
    secondChoice = str(var.get("featureModel"))
    fileName = models[int(choice) - 1]
    cleanedTweets = []
    for i in range(0, number_of_rows):
        tweet = inputData[i]
        tweet = re.sub("[^\u0600-\u06FF ]", '', tweet)  # Remove all special characters and numbers.
        tweet = re.sub(r'(.)\1+', r'\1', tweet)
        tweetWords = tweet.split()  # Split the words in the review.
        st = ISRIStemmer()  # Creating a stemmer object.
        cleanedTweet = [dediac_ar(word) if secondChoice == '2' or secondChoice == '4' else st.stem(dediac_ar(word))
                        for word in tweetWords if word not in set(stop_words)]
        cleanedTweet = u' '.join(
            cleanedTweet)  # Join all words together after stemming them and removing stopwords.
        cleanedTweets.append(cleanedTweet)
    vectorizer = CountVectorizer() if secondChoice >= '3' else TfidfVectorizer()
    data = vectorizer.fit_transform(cleanedTweets).toarray()
    tweet_train, tweet_test, offensive_train, offensive_test = train_test_split(data,
                                                                                dataSet.iloc[:, -1].values,
                                                                                test_size=0.2, random_state=0)
    if secondChoice == '1':
        fileName += "_TfIdf_with_stemming"
    elif secondChoice == '2':
        fileName += "_TfIdf_without_stemming"
    elif secondChoice == '3':
        fileName += "_BOW_with_stemming"
    else:
        fileName += "_BOW_without_stemming"

    if choice != '6':
        fileName += ".sav"

    if inputType == '1':
        res = load_model(fileName, 0, tweet_test, offensive_test)
        return json.dumps(res, cls=NpEncoder)

    else:
        cleanedTweets = []
        inputString = var.get("inputStr")
        tweet = inputString
        tweet = re.sub("[^\u0600-\u06FF ]", '', tweet)  # Remove all special characters and numbers.
        tweet = re.sub(r'(.)\1+', r'\1', tweet)
        tweetWords = tweet.split()  # Split the words in the review.
        st = ISRIStemmer()  # Creating a stemmer object.
        cleanedTweet = [dediac_ar(word) if secondChoice == '2' or secondChoice == '4' else st.stem(dediac_ar(word)) for
                        word in tweetWords if word not in set(stop_words)]
        cleanedTweet = u' '.join(cleanedTweet)  # Join all words together after stemming them and removing stopwords.
        cleanedTweets.append(cleanedTweet)
        data = vectorizer.transform(cleanedTweets).toarray()
        res = load_model(fileName, 1, tweet_test, offensive_test, data)
        return json.dumps(res, cls=NpEncoder)


if __name__ == "__main__":
    app.run(debug=True)
