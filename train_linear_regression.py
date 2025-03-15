import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Function to load data from a text file
def load_data_from_text_file():
    # Example sample data (CSV format) in a text file
    data = pd.read_csv('data.txt', delimiter=',')
    return data

# Function to load data from an SQLite database
def load_data_from_db():
    conn = sqlite3.connect('data.db')
    query = "SELECT * FROM data_table"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

# Train the model and save it
def train_and_save_model(data):
    # Assume the last column is the target (y) and others are features (X)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print('--------')

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the model to a file
    joblib.dump(model, 'linear_regression_model.pkl')

# Main function
def main():
    # Load data (choose source: file or database)
    data = None
    if os.path.exists('data1.txt'):
        data = load_data_from_text_file()
    elif os.path.exists('data.db'):
        data = load_data_from_db()

    if data is not None:
        # Train the model and save it
        train_and_save_model(data)
    else:
        print("No data found!")

if __name__ == "__main__":
    main()
