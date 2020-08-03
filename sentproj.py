from wtforms import Form, TextField, validators, SubmitField
from flask import Flask, render_template, request, jsonify, make_response
from utils import eval_from_input
from utils import load_model

#model = load_model()

#create app
app = Flask(__name__)

class WebForm(Form):
    eval_text = TextField("Please enter some text to be evaluated",
                          validators=[validators.InputRequired()])
    submit = SubmitField("Enter")

#@app.route("/", methods=['GET', 'POST'])
#def home():
#    "home page with form"
#    form = WebForm(request.form)
#
#    if request.method == 'POST' and form.validate():
#        eval_text = request.form['eval_text']
#        return render_template('eval_text.html',
#                               input=eval_from_input(eval_text=eval_text, model=model))
#
#    return render_template('index.html', form=form)



@app.route("/")
def home():
    return jsonify({'image_link':'test'}) #image_url})

@app.route("/evaltext", methods=['GET'])
def evaluate_text():#text):
    #eval_from_input(eval_text=eval_text, model=model))
    #image_url = 'test'
    return jsonify({'image_link':'test'}) #image_url})



#@app.route('/about/')
#def about():
#        return render_template('about.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
