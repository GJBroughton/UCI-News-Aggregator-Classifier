import pandas as pd
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
import string
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
import argparse
import os
import settings

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--text", required=False, help="manual imput of text. Provide string with quotes")
ap.add_argument("-f", "--file", help="path to csv file containing headlines to be classified.")
args = vars(ap.parse_args())

#load classifier
filename_model=filename_model=os.path.join(settings.MODEL_DIR,'uci_news_classifier.pkl')
with open(filename_model, 'rb') as f:
	loaded_model=pickle.load(f)

# to classify single given example
if args["text"]:
	text=str(args['text'])
	print(str(text))
	print(loaded_model.predict([text]))

# classify set of given examples saved as csv file
elif args["file"]:
	data=pd.read_csv(args["file"],encoding="ISO-8859-1")
	if data.shape[1]>1:
		print("csv file needs to contain one column with one headline per row")
	else:
		print(loaded_model.predict(data[data.columns[0]].tolist()))
		
else:
	print("Provide --text for single line or --file for csv")