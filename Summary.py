import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('customer_booking.csv', encoding='ISO-8859-1')

# Basic EDA and mapping based on the provided notebook
mapping = {
    "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, 
    "Fri": 5, "Sat": 6, "Sun": 7
}
df["flight_day"] = df["flight_day"].map(mapping)

# Check categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
print("Categorical columns:", cat_cols)

# We will use LabelEncoding for categorical columns with many unique values (like route, booking_origin)
# and One-Hot Encoding for the rest, but for simplicity with RandomForest, LabelEncoding works fine for all.
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop('booking_complete', axis=1)
y = df['booking_complete']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Train the model
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plotting Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importances - Random Forest Model')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nFeature Importances:")
print(feature_importance_df)