import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load your dataset (replace 'Explore.csv' with your actual dataset file)
file_path = 'data/Explore.csv'  # Adjust the path based on your folder structure in GitHub
df = pd.read_csv(file_path)

# Assuming your dataset has a 'College' column as the target variable

# Preprocess your data
features = df[['Rank', 'Gender', 'Category', 'Branch']]
target = df['College']

# Convert categorical variables into numerical ones using Label Encoding
label_encoder_gender = LabelEncoder()
label_encoder_category = LabelEncoder()
label_encoder_branch = LabelEncoder()

# Handle 'Gender' separately
features['Gender'] = label_encoder_gender.fit_transform(features['Gender'])
features['Category'] = label_encoder_category.fit_transform(features['Category'])
features['Branch'] = label_encoder_branch.fit_transform(features['Branch'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a decision tree classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model with the training set
dt_model.fit(X_train, y_train)

# Evaluate the accuracy on the test set
predictions = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
accuracy_percentage = accuracy * 100
print(f"Accuracy on the test set: {accuracy_percentage:.2f}%")

while True:
    # Take user input for features
    rank = int(input("Enter Rank: "))
    gender = input("Enter Gender (M/F): ")  # Modified to accept 'M' or 'F'
    category = input("Enter Category: ")
    branch = input("Enter Branch: ")

    # Convert user input using the same label encoders
    gender_encoded = label_encoder_gender.transform([gender])[0]
    category_encoded = label_encoder_category.transform([category])[0]
    branch_encoded = label_encoder_branch.transform([branch])[0]

    # Create a DataFrame with user input
    user_data = pd.DataFrame({'Rank': [rank], 'Gender': [gender_encoded], 'Category': [category_encoded], 'Branch': [branch_encoded]})

    # Make predictions for user input
    user_prediction = dt_model.predict(user_data)

    print(f"The predicted college for the given input is: {user_prediction[0]}")

    another_input = input("Do you want to enter another set of inputs? (yes/no): ")
    if another_input.lower() != 'yes':
        break
