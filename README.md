## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, PowerTransformer
df = pd.read_csv(r"C:\Users\acer\Downloads\data.csv") 
print(df.head()) 
```

<img width="942" height="176" alt="Screenshot 2025-10-16 103948" src="https://github.com/user-attachments/assets/51cccb99-0177-4ff1-aa28-cc72524ebe27" />

```
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Categorical Columns:", categorical_cols)  
print("Numerical Columns:", numerical_cols)
```

<img width="1227" height="64" alt="Screenshot 2025-10-16 104155" src="https://github.com/user-attachments/assets/3206c37a-a882-4d0e-af89-c21af9333177" /> 

```
labelencoder = LabelEncoder()
for col in categorical_cols:
    df[col] = labelencoder.fit_transform(df[col])
print("After Label Encoding:")
print(df.head()) 
```

<img width="1203" height="200" alt="Screenshot 2025-10-16 104410" src="https://github.com/user-attachments/assets/c82ad19f-85e1-43e9-ac76-3aa240165eac" />

```
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("After One-Hot Encoding:")
print(df_encoded.head()) 
```

<img width="1145" height="394" alt="Screenshot 2025-10-16 104531" src="https://github.com/user-attachments/assets/e56dce97-82e1-459d-95ed-d0198f6628cd" /> 

```
scaler = StandardScaler()
df_standardized = df_encoded.copy()
df_standardized[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
print("After Standardization:")
print(df_standardized.head()) 
```

<img width="1102" height="392" alt="Screenshot 2025-10-16 104704" src="https://github.com/user-attachments/assets/87f09c62-5dcf-4075-b8a2-f176bec9f4d5" /> 

```
minmax = MinMaxScaler()
df_normalized = df_encoded.copy()
df_normalized[numerical_cols] = minmax.fit_transform(df_encoded[numerical_cols])
print("After Normalization:")
print(df_normalized.head()) 
```

<img width="1319" height="394" alt="Screenshot 2025-10-16 104822" src="https://github.com/user-attachments/assets/72529add-77ca-4e6a-9066-314ff3d81973" /> 

```
pt = PowerTransformer(method='yeo-johnson')
df_transformed = df_encoded.copy()
df_transformed[numerical_cols] = pt.fit_transform(df_encoded[numerical_cols])
print("After Power Transformation:")
print(df_transformed.head()) 
```

<img width="1182" height="392" alt="Screenshot 2025-10-16 104953" src="https://github.com/user-attachments/assets/73c4ea79-3b5c-4e19-8b7f-7ea9027cd1a8" /> 


# RESULT:
       
 we perform Feature Encoding and Transformation process
       
