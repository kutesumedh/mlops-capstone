import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import mlflow
import mlflow.sklearn


def main():
    # Point MLflow to a local SQLite database for tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Create or use an experiment (instead of the default one)
    mlflow.set_experiment("housing_price_prediction")

    # Load dataset
    df = pd.read_csv("data/housing.csv")
    X = df[["area"]]   # Feature(s)
    y = df["price"]    # Target

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Dummy hyperparameters (just to log in MLflow)
    learning_rate = 0.01
    epochs = 100

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)

        # Train a simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate model
        r2 = model.score(X_test, y_test)
        mse = mean_squared_error(y_test, model.predict(X_test))

        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.2f}")

        # Log metrics
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)

        # Save trained model locally
        joblib.dump(model, "model.pkl")

        # Example input for MLflow model logging
        sample_input = pd.DataFrame({"area": [1000]})

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="linear_regression_model",
            input_example=sample_input
        )


if __name__ == "__main__":
    main()
