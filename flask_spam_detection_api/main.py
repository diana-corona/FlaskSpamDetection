from flask import request
from flask_spam_detection_api import app
from flask_spam_detection_api.classify_message import classify_message_function
from flask_spam_detection_api.load_model import load_model_function
from flask_spam_detection_api.preprocessor import preprocessor_function

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route("/query")
def hello():
	args = request.args
	message = args["message"]
	return f"{message}"

@app.route('/mlpclassifiers')
def mlpclassifiers():
	args = request.args
	message = args["message"]
	message=preprocessor_function(message)
	sel_model = load_model_function('MLPClassifier');
	return f"{classify_message_function(sel_model, message)}"

@app.route('/kneighbors')
def kneighbors():
	args = request.args
	message = args["message"]
	message=preprocessor_function(message)
	sel_model = load_model_function('KNeighbors');
	return classify_message_function(sel_model, message)

@app.route('/decisiontrees')
def decisiontrees():
	args = request.args
	message = args["message"]
	message=preprocessor_function(message)
	sel_model = load_model_function('DecisionTree');
	return classify_message_function(sel_model, message)

@app.route('/randomforests')
def randomforests():
	args = request.args
	message = args["message"]
	message=preprocessor_function(message)
	sel_model = load_model_function('RandomForest');
	return classify_message_function(sel_model, message)