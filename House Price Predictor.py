import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tkinter import Tk, Label, Entry, Button

# Load dataset
df = pd.read_csv("../../../Documents/MACHINE LEARNING/House Price Predictor/Housing.csv")

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Encode categorical variables (if any exist)
categorical_columns = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea',
    'furnishingstatus'
]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Define features (X) and target (y)
X = df.drop(columns=['price'])
y = df['price']

# Train-test split for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and fit the regression model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Model coefficients and intercept
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Model performance
score = reg.score(X_test, y_test)
print("Model R^2 Score:", score)

# Predict on the test set
y_pred = reg.predict(X_test)


# GUI for user input
def predict_gui():
    def predict():
        # Gather inputs from the GUI
        user_data = [
            float(area_entry.get()), int(bedrooms_entry.get()), int(bathrooms_entry.get()),
            int(stories_entry.get()), int(mainroad_entry.get()), int(guestroom_entry.get()),
            int(basement_entry.get()), int(hotwaterheating_entry.get()), int(airconditioning_entry.get()),
            int(parking_entry.get()), int(prefarea_entry.get())
        ]

        furnishingstatus = furnishingstatus_entry.get().lower()
        furnished_status_encoded = [0, 0]
        if furnishingstatus == "furnished":
            furnished_status_encoded = [1, 0]
        elif furnishingstatus == "semi-furnished":
            furnished_status_encoded = [0, 1]
        user_data += furnished_status_encoded

        # Make prediction
        predicted_price = reg.predict([user_data])[0]
        result_label.config(
            text=f"Predicted Price: {predicted_price:,.2f}"
        )

        # Plot the actual vs. predicted graph with the user's input
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label="Predictions")
        plt.scatter(
            [predicted_price], [predicted_price],
            color='green', s=100, label="Your Prediction"
        )
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color='red', linestyle='--', linewidth=2
        )
        plt.title("Actual vs Predicted Prices")
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Create the GUI window
    root = Tk()
    root.title("House Price Prediction")

    # Input fields
    Label(root, text="Area (in square feet):").grid(row=0, column=0)
    area_entry = Entry(root)
    area_entry.grid(row=0, column=1)

    Label(root, text="Number of bedrooms:").grid(row=1, column=0)
    bedrooms_entry = Entry(root)
    bedrooms_entry.grid(row=1, column=1)

    Label(root, text="Number of bathrooms:").grid(row=2, column=0)
    bathrooms_entry = Entry(root)
    bathrooms_entry.grid(row=2, column=1)

    Label(root, text="Number of stories:").grid(row=3, column=0)
    stories_entry = Entry(root)
    stories_entry.grid(row=3, column=1)

    Label(root, text="Main road (1 for Yes, 0 for No):").grid(row=4, column=0)
    mainroad_entry = Entry(root)
    mainroad_entry.grid(row=4, column=1)

    Label(root, text="Guest room (1 for Yes, 0 for No):").grid(row=5, column=0)
    guestroom_entry = Entry(root)
    guestroom_entry.grid(row=5, column=1)

    Label(root, text="Basement (1 for Yes, 0 for No):").grid(row=6, column=0)
    basement_entry = Entry(root)
    basement_entry.grid(row=6, column=1)

    Label(root, text="Hot water heating (1 for Yes, 0 for No):").grid(row=7, column=0)
    hotwaterheating_entry = Entry(root)
    hotwaterheating_entry.grid(row=7, column=1)

    Label(root, text="Air conditioning (1 for Yes, 0 for No):").grid(row=8, column=0)
    airconditioning_entry = Entry(root)
    airconditioning_entry.grid(row=8, column=1)

    Label(root, text="Number of parking spaces:").grid(row=9, column=0)
    parking_entry = Entry(root)
    parking_entry.grid(row=9, column=1)

    Label(root, text="Preferred area (1 for Yes, 0 for No):").grid(row=10, column=0)
    prefarea_entry = Entry(root)
    prefarea_entry.grid(row=10, column=1)

    Label(root, text="Furnishing status (semi-furnished, furnished, unfurnished):").grid(row=11, column=0)
    furnishingstatus_entry = Entry(root)
    furnishingstatus_entry.grid(row=11, column=1)

    # Prediction result
    result_label = Label(root, text="")
    result_label.grid(row=12, columnspan=2)

    # Predict button
    Button(root, text="Predict", command=predict).grid(row=13, columnspan=2)

    root.mainloop()


# Call the GUI
predict_gui()
