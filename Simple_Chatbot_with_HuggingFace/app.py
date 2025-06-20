from flask import Flask
from flask_cors import CORS
from bot import chatbot
from flask import request, render_template
import json

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    return chatbot.chat(input_text)

if __name__ == '__main__':
    app.run()