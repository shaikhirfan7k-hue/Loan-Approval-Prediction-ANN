"""Train an ANN model for loan approval prediction from loan_data.csv."""
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_FILE = "loan_data.csv"
MODEL_FILE = "loan_ann_model.keras"
PREPROCESSOR_FILE = "loan_preprocessor.pkl"

CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "Loan_Status"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Place the dataset in this project folder.")

df = pd.read_csv(DATA_FILE)
missing_features = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_features:
    raise ValueError("Dataset is missing columns: " + ", ".join(missing_features))
df = df[FEATURES + [TARGET]].copy()
df = df.dropna(subset=[TARGET])
y = df[TARGET].map({"Y": 1, "N": 0})
X = df[FEATURES]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
preprocessor = ColumnTransformer([
    ("categorical", cat_pipe, CATEGORICAL),
    ("numeric", num_pipe, NUMERIC),
])
X_train_ready = preprocessor.fit_transform(X_train)
X_test_ready = preprocessor.transform(X_test)
if hasattr(X_train_ready, "toarray"):
    X_train_ready = X_train_ready.toarray()
    X_test_ready = X_test_ready.toarray()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_ready.shape[1],)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train_ready, y_train, validation_split=0.2, epochs=50, batch_size=16,
    verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )]
)
probabilities = model.predict(X_test_ready, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)
print("Test Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions, zero_division=0))
model.save(MODEL_FILE)
joblib.dump(preprocessor, PREPROCESSOR_FILE)
print(f"Saved {MODEL_FILE} and {PREPROCESSOR_FILE}")

plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("ANN Training and Validation Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("accuracy_graph.png", dpi=150)
plt.show()
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("ANN Training and Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("loss_graph.png", dpi=150)
plt.show()
