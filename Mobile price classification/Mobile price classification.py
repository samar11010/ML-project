import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load and prepare full dataset
df = pd.read_csv("Data/Original data/excel1.csv")
X_full = df.drop(columns=["price_range"])
y_full = df["price_range"]

# Impute and scale all features
imputer_all = SimpleImputer(strategy="mean")
X_full_imputed = imputer_all.fit_transform(X_full)

scaler_all = StandardScaler()
X_full_scaled = scaler_all.fit_transform(X_full_imputed)

# Fit RandomForest to all features for importance analysis
rf = RandomForestClassifier(random_state=42)
rf.fit(X_full_scaled, y_full)

# Extract top 7 features based on importance
importances = rf.feature_importances_
feature_names = X_full.columns
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
top_features = importance_df["Feature"].head(7).tolist()
df[top_features].describe().to_csv("top_features_description.csv")

print("Top 7 important features selected:")
print(top_features)

# Use top features for modeling
X = df[top_features]
y = df["price_range"]

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Balance training data using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Save preprocessed data
pd.DataFrame(X_train_balanced, columns=top_features).to_csv("Data/Preprocessed data/X_train.csv", index=False)
pd.DataFrame(X_test, columns=top_features).to_csv("Data/Preprocessed data/X_test.csv", index=False)
y_train_balanced.to_csv("Data/Preprocessed data/Y_train.csv", index=False)
y_test.to_csv("Data/Preprocessed data/Y_test.csv", index=False)

# Define models
models = {
    "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True),
    "NaiveBayes": GaussianNB(),
    "ANN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# Train and evaluate
accuracies = {}
labels = ['0', '1', '2', '3']  # for price_range classes

for name, model in models.items():
    print(f"\n** Training {name} **")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)

    pd.DataFrame({"Prediction": y_pred}).to_csv(f"Data/Results/prediction_{name}.csv", index=False)

    print(classification_report(y_test, y_pred, zero_division=0))

    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for {name}\n(Shows Correct vs Incorrect Predictions)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'Data/Results/confusion_matrix_{name}.png')
    plt.show()

# Accuracy comparison bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.title("Model Accuracy Comparison\n(Overall Classification Accuracy on Test Set)")
plt.ylabel("Accuracy Score")
plt.xlabel("Model")
plt.ylim(0.0, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Data/Results/model_accuracy_comparison.png")
plt.show()
