from flask import Flask, request, render_template
from mymodel import *
from utils import *

app = Flask(__name__)
# Initialize your HateDetector model
HateDetector = CustomModel('models/binary', 'models/multi')


# Function to analyze sentiment
def analyze_sentiment(user_input):
    if not user_input:
        return user_input, "Please enter a sentence."
    
    logits_list = HateDetector.inference(user_input)
    probabilities_list = softmax_grouped(logits_list) 
    return probabilities_list


@app.route('/', methods=['GET', 'POST'])
def index():
    probabilities = None
    user_input = None
    labels = ["Non-Hate", "Hate", "Acceptable", "Inappropriate", "Offensive", "Violent"]

    if request.method == 'POST':
        user_input = request.form['comment']
        probabilities = analyze_sentiment(user_input)
    
    return render_template('index.html', user_input=user_input, probabilities=probabilities, labels=labels, zip=zip)

if __name__ == '__main__':
    app.run(debug=True)



