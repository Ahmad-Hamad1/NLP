import pickle
import sys
import nltk
import numpy as np
import pandas as pd
import regex as re
from camel_tools.utils.dediac import dediac_ar
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras

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
outputClass = dataSet.iloc[:, 1].values
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit([outputClass])
outputClass = imp.transform([outputClass])

number_of_rows = len(inputData)
stopWords = nltk.download("stopwords")
stop_words = stopwords.words('arabic')

for idx, word in enumerate(stop_words):
    stop_words[idx] = dediac_ar(word)


def print_confusion_matrix(y_test, spam_prediction, string):
    cm = confusion_matrix(y_test, spam_prediction)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    pr = tp / (tp + fp)
    rec = tp / (tp + fn)
    print("\n========= The Confusion Matrix of", string, "========")
    print("                 True Positive  =", tp)
    print("                 False Positive =", fp)
    print("                 False Negative =", fn)
    print("                 True Negative  =", tn)
    print("                 Precision =", round(pr * 100, 3), "%")
    print("                 Recall    =", round(rec * 100, 3), "%")
    print("                 F1-score  =", round((2 * pr * rec) / (pr + rec) * 100, 3), "%")
    print("                 Accuracy  =", round(accuracy_score(y_test, spam_prediction) * 100, 3), "%")


def train_model(name_of_model):
    if name_of_model == models[0]:
        classifier_DT = DecisionTreeClassifier(criterion='entropy', random_state=0)
        classifier_DT.fit(tweet_train, offensive_train)
        offensive_prediction_DT = classifier_DT.predict(tweet_test)
        print_confusion_matrix(offensive_test, offensive_prediction_DT, name_of_model)
    elif name_of_model == models[1]:
        classifier_RF = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier_RF.fit(tweet_train, offensive_train)
        offensive_prediction_RF = classifier_RF.predict(tweet_test)
        print_confusion_matrix(offensive_test, offensive_prediction_RF, name_of_model)
    elif name_of_model == models[2]:
        classifier_GNB = GaussianNB()
        classifier_GNB.fit(tweet_train, offensive_train)
        offensive_prediction_GNB = classifier_GNB.predict(tweet_test)
        print_confusion_matrix(offensive_test, offensive_prediction_GNB, name_of_model)
    elif name_of_model == models[3]:
        classifier_SVC = SVC(kernel='linear', random_state=0)
        classifier_SVC.fit(tweet_train, offensive_train)
        offensive_prediction_SVC = classifier_SVC.predict(tweet_test)
        print_confusion_matrix(offensive_test, offensive_prediction_SVC, name_of_model)
    elif name_of_model == models[4]:
        classifier_KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        classifier_KNN.fit(tweet_train, offensive_train)
        offensive_prediction_KNN = classifier_KNN.predict(tweet_test)
        print_confusion_matrix(offensive_test, offensive_prediction_KNN, name_of_model)


def load_model(file_name, name_of_model, run_type):
    if file_name.startswith("ANN"):
        reconstructed_model = keras.models.load_model(file_name)
        offensive_prediction = reconstructed_model.predict(tweet_test if run_type == 0 else data) > 0.5
        if run_type == 0:
            print_confusion_matrix(offensive_test, offensive_prediction, name_of_model)
        else:
            if offensive_prediction:
                print("The given string is offensive")
            else:
                print("The given string is not offensive")
        return

    loaded_model = pickle.load(open(file_name, 'rb'))
    offensive_prediction = loaded_model.predict(tweet_test if run_type == 0 else data)
    if run_type == 0:
        print_confusion_matrix(offensive_test, offensive_prediction, name_of_model)
    else:
        if offensive_prediction == 1.0:
            print("The given string is offensive")
        else:
            print("The given string is not offensive")


# if __name__ == "__main__":
while True:
    sys.stdout.flush()
    print("\nEither choose a model or exit:")
    print("1) Decision_Tree")
    print("2) Random_Forest")
    print("3) Naive_Bayes")
    print("4) Support_Vector_Machine")
    print("5) KNN model ")
    print("6) Artificial Neural network")
    print("7) Exit\n")

    choice = re.sub("[^1-7]", "", str(input("Enter a number: ")))

    if len(choice) == 0:
        print("Please enter a valid number")
        continue

    if choice == "7":
        exit(0)

    print("\nChose the mode of operation")
    print("\n1) To train and test the model on the data set \n2) To run the model on a uer-input string")
    inputType = input("Chose the type : ")

    print("\nChose feature model")
    print("1) TF-IDF with stemmer \n2) TF-IDF without stemmer \n3) Bag of Words with stemmer \n4) Bag of Words "
          "without stemmer")
    secondChoice = input("\nEnter a number: ")
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
        load_model(fileName, models[int(choice) - 1], 0)
    else:
        cleanedTweets = []
        inputString = input("\nEnter the string to test: ")
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
        load_model(fileName, models[int(choice) - 1], 1)
