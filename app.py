from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

regr_model = joblib.load('./notebooks/linear_regr.pkl')
tree_model = joblib.load('./notebooks/tree.pkl')

@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        print(request.form)
        beds = int(request.form['beds'])
        baths = float(request.form['baths'])
        sqft = int(request.form['sqft'])
        age = int(request.form['age'])
        lotsize = float(request.form['lotsize'])
        garage = int(request.form['garage'])
        model = request.form['model']
    else:
        # Make prediction - features = ['BEDS', 'BATHS', 'SQFT', 'AGE', 'LOTSIZE', 'GARAGE']
        beds = 4
        baths = 2.5
        sqft = 3005
        age = 15
        lotsize = 17903.0
        garage = 1
        model = "linear"
    features = [[beds, baths, sqft, age, lotsize, garage]]
    print(features)
    # features = [[4, 2.5, 3005, 15, 17903.0, 1]]
    if model == "linear":
        prediction = int(regr_model.predict(features)[0])
    elif model == "tree":
        prediction = int(tree_model.predict(features)[0])
    else:
        print("Error - no model type selected. Using linear regression.")
        prediction = regr_model.predict(features)[0][0].round(1)
    # print(regr_model.predict([[4,1,1536,138,8000,2]])[0][0])
    # tree_prediction = tree_model.predict(features)[0][0].round(1)
    return render_template('index.html', beds=beds, baths=baths, sqft=sqft, age=age, lotsize=lotsize, garage=garage, prediction=prediction,model=model)
