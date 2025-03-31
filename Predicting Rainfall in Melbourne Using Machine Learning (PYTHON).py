#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


# In[2]:


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()


# In[3]:


df.count()


# In[4]:


#drop all rows with missing value
df = df.dropna()
df.info()


# In[6]:


df.columns


# In[7]:


df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })


# In[8]:


df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()


# In[9]:


def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'


# In[11]:


# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply the function to the 'Date' column
df['Season'] = df['Date'].apply(date_to_season)

# Drop the 'Date' column as it's no longer needed
df = df.drop(columns=['Date'])

# Display the updated dataframe
df.head()


# In[12]:


# Define feature dataframe (X) by dropping the target column
X = df.drop(columns=['RainToday'], axis=1)

# Define target variable (y)
y = df['RainToday']

# Display shapes of X and y to verify
print(X.shape, y.shape)


# In[13]:


# Check class distribution in the target variable
y.value_counts()


# Exercise 4: Analysis of Class Distribution
# -How often does it rain annually in Melbourne?
# 
# There are 1,791 "Yes" (rainy days) and 5,766 "No" (dry days) in the dataset.
# 
# Proportion of rainy days:
# 
# 11791/(1791+5766)
#  ≈23.7%
# Conclusion: It rains about 24% of the time in the Melbourne area.
# 
#  
#  -How accurate would I be if I always predicted "No Rain"?
# 
# If you always predicted "No", your accuracy would be:
# 
# 5766/(5766+1791)≈76.3%
# Conclusion: A simple "No Rain" classifier would be 76.3% accurate
# 
# 
# -Is this dataset balanced?
# 
# No, it's imbalanced. The dataset is skewed towards "No Rain" (76.3% vs. 23.7%).
# 
# 
# 

# In[15]:


from sklearn.model_selection import train_test_split

# Split the data into training and test sets (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Display class distribution in training and test sets
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


# In[45]:


# Identify numerical and categorical features
numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()  
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Display the detected features
print("Numeric Features:", numeric_features)
print("Categorical Features:", categorical_features)


# In[46]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Define transformations
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),  # Apply StandardScaler to numeric columns
        ('cat', categorical_transformer, categorical_features)  # Apply OneHotEncoder to categorical columns
    ]
)

# Display the preprocessor
print(preprocessor)


# In[48]:


# Create a pipeline with preprocessing and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Apply preprocessing (scaling + encoding)
    ('classifier', RandomForestClassifier(random_state=42))  # Train a Random Forest model
])

# Display the pipeline
print(pipeline)


# In[49]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Define parameter grid for tuning Random Forest
param_grid = {
    'classifier__n_estimators': [50, 100],  # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20],  # Depth of trees
    'classifier__min_samples_split': [2, 5]  # Minimum samples required to split a node
}

# Define cross-validation strategy (Stratified K-Fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Instantiate GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Display the best parameters
print("Best Parameters:", grid_search.best_params_)

# Display the best score
print("Best Cross-Validation Accuracy:", grid_search.best_score_)


# In[50]:


# Evaluate the best model on the test set
test_score = grid_search.score(X_test, y_test)  

# Print the test set accuracy
print("Test set score: {:.2f}".format(test_score))


# In[52]:


# Evaluate the best model on the test set
test_score = grid_search.score(X_test, y_test)  

# Print the test set accuracy
print("Test set score: {:.2f}".format(test_score))


# In[53]:


# Get model predictions on the test set
y_pred = grid_search.predict(X_test)

# Display first few predictions
print(y_pred[:10])  # Show first 10 predictions


# In[54]:


from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[55]:


# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a ConfusionMatrixDisplay instance
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)

# Plot the confusion matrix
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# In[56]:


# Extract feature importances from the best model
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

# Display the feature importances
print("Feature Importances:", feature_importances)


# In[59]:


import matplotlib.pyplot as plt
import pandas as pd

# Combine numeric and categorical feature names
categorical_feature_names = grid_search.best_estimator_['preprocessor'] \
                                .named_transformers_['cat'] \
                                .get_feature_names_out(categorical_features)

# Combine numeric features with the one-hot encoded categorical feature names
feature_names = numeric_features + list(categorical_feature_names)

# Extract feature importances from the best model
feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

# Create a DataFrame to hold feature names and their importances
importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Select top N features (change N to display more or fewer)
N = 20
top_features = importance_df.head(N)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()


# In[60]:


# Replace RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# update the model's estimator to use the new pipeline
grid_search.estimator = pipeline

# Define a new grid with Logistic Regression parameters
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

grid_search.param_grid = param_grid

# Fit the updated pipeline with LogisticRegression
grid_search.fit(X_train, y_train)

# Make predictions
y_pred = grid_search.predict(X_test)


# In[61]:


#comparing results to the previous model
print(classification_report(y_test, y_pred))

# Generate the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Set the title and labels
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Show the plot
plt.tight_layout()
plt.show()


# ##Summary
# 
# This project focuses on building a predictive model to forecast rainfall in Melbourne based on historical weather data. The dataset includes various weather attributes, such as temperature, humidity, wind speed, and pressure, for different locations in Melbourne. The data preprocessing steps involved cleaning the dataset by handling missing values and converting the `Date` column into seasons. The target variable, originally "RainTomorrow," was renamed to "RainToday" to better reflect the prediction goal, and the dataset was further filtered to include only specific locations. The data was then split into training and testing sets using stratified sampling to maintain the class distribution.
# 
# The project implemented machine learning models, primarily focusing on Random Forest and Logistic Regression classifiers. Feature engineering was employed to handle both numeric and categorical features, where the numeric features were standardized and categorical features were one-hot encoded. Hyperparameter tuning was conducted using GridSearchCV with cross-validation to optimize model performance. The results showed that both models provided valuable insights, but the Logistic Regression model, after parameter tuning, was especially effective in balancing the trade-off between precision and recall for the imbalanced target variable.
# 
# In the final steps, feature importance was extracted from the best-performing model, providing valuable insights into which weather attributes had the most significant impact on predicting rainfall. The project concluded by comparing the performance of both models using metrics such as accuracy, confusion matrix, and classification reports. This analysis can be extended further for real-time predictions or applied to similar predictive tasks in meteorology, offering a robust approach to weather forecasting using machine learning techniques.

# In[ ]:




