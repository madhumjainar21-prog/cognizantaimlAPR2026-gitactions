import os
from dotenv import load_dotenv
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

def fraud_detection_train(file_path):
    """
    Train a fraud detection model using the provided dataset.

    Parameters:
    file_path (str): The path to the CSV file containing the training data.

    Returns:
    model: The trained fraud detection model.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    df.dropna(inplace=True) 
    df = pd.get_dummies(df, columns=['location'],drop_first=True)
    x=df.drop('label',axis=1)
    y=df['label']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = (y_test == y_pred).mean()
    print(f"Model accuracy: {accuracy:.2f}")

    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.2f}")
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    os.makedirs("outputs", exist_ok=True)
 
    model_path = os.getenv('model_path')
 
    artifact = {
            "model": clf,
            "feature_columns": x.columns.tolist(),
            "metrics": {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            }
        }
 
 
    joblib.dump(artifact, model_path)
 
    print(f"\nModel saved to: {model_path}")
 
    return model_path
 

if __name__ == "__main__":
    # Example usage
    file_path = os.getenv("file_path")
    fraud_detection_train(file_path)

