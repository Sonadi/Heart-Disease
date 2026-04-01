import pandas as pd
import numpy as np

# Step 1: Column names
columns = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal","target"
]

# Step 2: Load dataset
df = pd.read_csv("processed.cleveland.data", names=columns)

print("Dataset loaded!")

# Step 3: Handle missing values
df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric)

# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

print("Missing values handled!")

# Step 4: Convert target to binary
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

print("Target converted!")

# Step 5: Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Step 6: Combine again (for saving)
df_final = pd.concat([X, y], axis=1)

# Step 7: Split train & test
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df_final,
    test_size=0.2,
    random_state=42
)

# Step 8: Save files
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("✅ Train and Test datasets created successfully!")