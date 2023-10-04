# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Load the Melbourne Housing dataset (replace 'melb_data.csv' with your dataset path)
            melb_data = pd.read_csv("melb_data1.csv")

            # Data preprocessing (select features and target variable)
            X = melb_data[["Bedroom", "Bathroom", "Landsize"]]
            y = melb_data["Price"]

            X = X.dropna()  # Remove rows with missing values
            y = y[X.index]  # Update the target variable accordingly


            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict the price
            rooms = float(request.form["rooms"])
            bathrooms = float(request.form["bathrooms"])
            landsize = float(request.form["landsize"])
            prediction = model.predict([[rooms, bathrooms, landsize]])

            return render_template("index.html", prediction=prediction[0])
        except Exception as e:
            return str(e)
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
