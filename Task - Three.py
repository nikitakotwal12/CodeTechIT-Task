from flask import Flask, request, jsonify
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# 1. Data Collection and Preprocessing
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split data into training and testing sets
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Model Building and Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 3. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# 4. API Development using Flask
app = Flask(_name_)

@app.route('/')
def home():
    return "Welcome to the Iris Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Get JSON input and convert to DataFrame
    data = request.get_json()
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)
    response = {
        "prediction": int(prediction[0]),
        "class_name": data.target_names[int(prediction[0])]
    }

    return jsonify(response)

if _name_ == '_main_':
    app.run(debug=True)
