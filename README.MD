# Classifying News Headlines from UCI ML News Aggregator Dataset

A simple classifier, classifying news headlines from the UCI ML News Aggregator dataset based on a Logistic Regression classifier from the scikit-learn library.

The data is the UCI News Aggregator dataset (https://archive.ics.uci.edu/ml/datasets/News+Aggregator) which contains headlines and classes for over 400,000 news articles.


### Installation:

 - git clone repository
 - pip install all python packages in 'requirements.txt'.

### Usage:

train.py
- open cmd prompt
- run python train.py to train model based on data provided in data directory

classify.py
- To classify single line example headline:
	- run python classify.py --text "Text to Classify"
- To classify multiple examples:
	- save examples as a csv file with each headline given in a single row
	- run python classify.py --file "path to csv file"


