import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load data
print("Loading data...")
df = pd.read_csv('data/emails.csv')

# 2. Separate features (X) and label (y)
X = df.drop(columns=['Email No.', 'Prediction'])
y = df['Prediction']

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# 3. Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 4. Train the model
print("\nTraining model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, 
      target_names=['Safe', 'Phishing']))

# 6. Save the model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/phishing_model.pkl')
print("\nModel saved to models/phishing_model.pkl")