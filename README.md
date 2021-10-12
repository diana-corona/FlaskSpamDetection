# FlaskSpamDetection
### Spam Text Message Classification Data
Obtained from

https://www.kaggle.com/team-ai/spam-text-message-classification

### Set a virtual enviroment and install requirements
```
virtualenv venv --python=python3
source venv/bin/activate
pip install  -r flask_spam_detection_api/requirements.txt

```

### Run flask 
```
export FLASK_APP=flask_spam_detection_api
flask run
```

### Query Parameters
Query parameters are part of the URL string and are prefixed by a “?”

For example:

http://localhost/spam_detection_query/?message=WINNER

### Data preprocesing  
The data was preprocessed by 
1. Removing HTML markup tags
1. Removing punctuation marks
1. Removing capital letters

### Models accuracy 
The data was randomly separed in using sklearn train_test_split.
80% of the data was used to train 
20% of the data was used to test the model and get the accuracy

1. MLPClassifier has an accuracy of 98.29%
1. KNeighbors has an accuracy of 93.36%
1. DecisionTreeClassifier has an accuracy of 95.96%
1. RandomForestClassifier has an accuracy of 97.04%
