import csv
import numpy as np

messages = []
labels = []

# Read the CSV file
with open("messages.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        messages.append(row["message"])
        labels.append(row["label"])


# Spam words that we designed ourselves
spam_words = [
    "prize",
    "winner",
    "win",
    "free",
    "reward",
    "offer",
    "bonus",
    "claim",
    "discount",
    "lucky",
    "urgent",
    "cash",
    "gift",
    "loan"
]


# Feature extraction function
def extract_features(message):

    # Feature 1: message length
    message_length = len(message)

    # Feature 2: number of digits
    digit_count = sum(char.isdigit() for char in message)

    # Feature 3: number of spam words
    words = message.lower().split()

    spam_word_count = 0

    for word in spam_words:
        if word in words:
            spam_word_count += 1

    # Feature 4: link presence
    if "http" in message.lower() or "www." in message.lower():
        has_link = 1
    else:
        has_link = 0

    # Feature 5: phone number presence
    phone_digits = sum(char.isdigit() for char in message)

    if phone_digits >= 10:
        has_phone = 1
    else:
        has_phone = 0

    # Feature 6: uppercase letters
    caps_count = sum(char.isupper() for char in message)

    return [
        message_length,
        digit_count,
        spam_word_count,
        has_link,
        has_phone,
        caps_count
    ]


# Create features for all messages
features = []

for message in messages:
    features.append(extract_features(message))


# Convert features to NumPy array
X = np.array(features)

print("Feature matrix shape:", X.shape)

print("\nFirst message:")
print(messages[0])

print("\nFeatures of first message:")
print(X[0])

print("\nSpam messages:", labels.count("spam"))
print("Not-spam messages:", labels.count("not-spam"))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Convert labels into numbers
y = np.array([1 if label == "spam" else 0 for label in labels])

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed!")

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# 5 brand-new messages written by the student
new_messages = [
    "Congratulations! Aap ne 75000 rupees ka prize jeeta hai.",
    "Kal university mein 10 baje milna hai.",
    "URGENT! Apka free reward claim karne ke liye abhi contact karein.",
    "Please mujhe Python assignment ki file bhej dena.",
    "Aap lucky winner hain, apna cash bonus abhi claim karein."
]

# Extract features from new messages
new_features = []

for message in new_messages:
    new_features.append(extract_features(message))

new_features = np.array(new_features)

# Predict
new_predictions = model.predict(new_features)

print("\n--- New Message Predictions ---")

for message, prediction in zip(new_messages, new_predictions):

    if prediction == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    print("\nMessage:", message)
    print("Prediction:", result)