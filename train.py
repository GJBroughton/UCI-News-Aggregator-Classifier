import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
import string
import pickle
import dill as pickle
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import datetime
import os
import settings

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

if __name__ == "__main__":
	now=datetime.datetime.now()
	time=now.strftime("%Y%m%d-%H%M")
	print("Reading training data...")
	data=pd.read_csv(os.path.join(settings.DATA_DIR, "uci-news-aggregator.csv"),encoding="ISO-8859-1")
	
	data['CATEGORY']=data['CATEGORY'].replace(['e','b','t','m'],[0,1,2,3])
	
	list_corpus = data['TITLE'].tolist()
	list_labels = data['CATEGORY'].tolist()
	X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)
	
	clf_pipeline = Pipeline([("vectorizer", TfidfVectorizer(tokenizer=LemmaTokenizer(),ngram_range=(1,3),stop_words='english')),
	 	("classifier", LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',multi_class='multinomial', n_jobs=-1, random_state=40))
		])
						
	print("Fitting model...")
	clf=clf_pipeline.fit(X_train, y_train)
	print("Classifying test set...\n")
	predictions = clf.predict(X_test)
	print("Classification Report:")
	print(classification_report(y_test, predictions, target_names=['e','b','t','m'])+"\n")
	with open("training_log.txt", "a") as fc: 
		fc.write("Date "+str(time)+"\n"+str(classification_report(y_test, predictions, target_names=['e','b','t','m']))+"\n\n")
	print("Serialising model...")
	filename_model=os.path.join(settings.MODEL_DIR,'uci_news_classifier.pkl')
	with open(filename_model, 'wb') as file:
		pickle.dump(clf ,file)
	print("Model Saved. Log updated.")