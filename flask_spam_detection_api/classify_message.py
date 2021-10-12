import joblib
import sklearn
from flask_spam_detection_api.preprocessor import preprocessor_function

def classify_message_function(model, message):
	message = preprocessor_function(message)
	#preprocess message like during the training
	label = model.predict([message])[0]
	#predict
	spam_prob = model.predict_proba([message])
	#calculate probability
	return {'label': label, 'spam_probability': spam_prob[0][1],'message':message}