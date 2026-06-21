import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

# Load dataset
df = pd.read_excel(
    "Comprehensive_MultiSector_Stock_Prediction_Data.xlsx",
    sheet_name="All_Historical_Data"
)

# Create target (next day's close price)
df["Target"] = df["Close"].shift(-1)

# Remove last row
df.dropna(inplace=True)

# Features
X = df[["Open", "High", "Low", "Volume"]]

# Target
y = df["Target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)
print("R2 Score:", score)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")