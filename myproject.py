from utils import eval_from_input, load_model

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
app = Flask(__name__)
api = Api(app)

model = load_model()

@app.route('/<string:text>', methods=['GET'])
def get_task(text):
    model_output = eval_from_input(eval_text=text, model=model)
    return jsonify({'image': model_output})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
