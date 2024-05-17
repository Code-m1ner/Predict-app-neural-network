from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Sample music data
music = pd.DataFrame({
    'Age': [
        20, 23, 25, 26, 29, 30, 31, 33, 37, 38,
        20, 21, 25, 26, 27, 30, 31, 34, 35, 38
    ],
    'Gender': [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ],
    'Genre': [
        'HipHop', 'HipHop', 'HipHop', 'Jazz', 'Jazz', 'Jazz', 'Classical', 'Classical', 'Classical', 'Dance',
        'Dance', 'Dance', 'Acoustic', 'Acoustic', 'Acoustic', 'Classical', 'Classical', 'Classical', 'Pop', 'Pop'
    ]
})

# Split data into features (X) and target (y)
X = music.drop(columns=['Genre'])
y = music['Genre']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_age = int(request.form['age'])
        user_gender = int(request.form['gender'])

        user_data = pd.DataFrame({'Age': [user_age], 'Gender': [user_gender]})
        user_prediction = model.predict(user_data)

        user_accuracy = accuracy_score([user_prediction[0]], model.predict(user_data))

        return redirect(url_for('result', prediction=user_prediction[0], accuracy=user_accuracy))

    return render_template('index.html')

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    accuracy = request.args.get('accuracy')
    return render_template('result.html', prediction=prediction, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)