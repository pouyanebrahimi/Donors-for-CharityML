
# coding: utf-8

# ## Supervised Learning
# ## Project: Finding Donors for *CharityML*

# ## Getting Started
# 
# In this project, several supervised algorithms are emplyed to accurately model individuals' income using data collected from the 1994 U.S. Census. Then the best candidate algorithm from preliminary results is found and is optimized more to best model the data. 
# 
# The goal is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can infer this value from other publically available features. 

# ----
# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the census data. 
# 
# Note that the last column from this dataset, `'income'`, will be the target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database.

# In[21]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))


# ### Implementation: Data Exploration
# A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \$50,000. In the code cell below, we will find:
# - The total number of records, `'n_records'`
# - The number of individuals making more than \$50,000 annually, `'n_greater_50k'`.
# - The number of individuals making at most \$50,000 annually, `'n_at_most_50k'`.
# - The percentage of individuals making more than \$50,000 annually, `'greater_percent'`.

# In[22]:


data.sample(n=5)


# In[23]:


# Total number of records
n_records = data.shape[0]

# Number of records where individual's income is more than $50,000
n_greater_50k = data.query('income==">50K"').shape[0]

# Number of records where individual's income is at most $50,000
n_at_most_50k = data.query('income=="<=50K"').shape[0]

# Other method
#n_at_most_50k, n_greater_50k = data.income.value_counts()

# Percentage of individuals whose income is more than $50,000
greater_percent = n_greater_50k*100/n_records

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))


# ** Featureset Exploration **
# 
# * **age**: continuous. 
# * **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
# * **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
# * **education-num**: continuous. 
# * **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
# * **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
# * **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
# * **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. 
# * **sex**: Female, Male. 
# * **capital-gain**: continuous. 
# * **capital-loss**: continuous. 
# * **hours-per-week**: continuous. 
# * **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# ----
# ## Preparing the Data
# 
# **Preprocessing:**
# Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as *preprocessing*. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

# In[24]:


# First let's split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)


# ### Transforming Skewed Continuous Features
# A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number. 
# 
# Run the code cell below to plot a histogram of these features. Note the range of the values present and how they are distributed.

# In[25]:


# Visualize skewed continuous features of original data
vs.distribution(data)


# Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. 
# 
# For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.
# 
# Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed. 

# In[26]:


# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x+1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)


# ### Normalizing Numerical Features
# In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.
# 
# Run the code cell below to normalize each numerical feature. We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this.

# In[27]:


# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))


# ### Implementation: Data Preprocessing
# 
# From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.
# 
# |   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |
# | :-: | :-: |                            | :-: | :-: | :-: |
# | 0 |  B  |  | 0 | 1 | 0 |
# | 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
# | 2 |  A  |  | 1 | 0 | 0 |
# 
# Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively. 

# In[28]:


# TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Encode the 'income_raw' data to numerical values
salary = {'<=50K': 0,'>50K': 1} 
income = [salary[i] for i in income_raw]
income=pd.DataFrame(income)
# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))


# In[29]:


# encoded feature names
print (encoded)


# ### Shuffle and Split Data
# Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. We will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.
# 
# Run the code cell below to perform this split.

# In[30]:


# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ----
# ## Evaluating Model Performance
# In this section, I will investigate four different algorithms, and determine which is best at modeling the data.

# ### Metrics and the Naive Predictor
# *CharityML*, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that *does not* make more than \$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:
# 
# $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
# 
# In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).
# 
# Looking at the distribution of classes (those who make at most \$50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \$50,000"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, *CharityML* would identify no one as donors. 

# ### Question 1 - Naive Predictor Performace
# * If we chose a model that always predicted an individual made more than $50,000, what would  that model's accuracy and F-score be on this dataset? 

# In[31]:


'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
# TODO: Calculate accuracy, precision and recall
total_data=income[0].count();
TP=np.sum(income[0]);
FP = total_data-TP;
TN = 0;
FN = 0; 

accuracy = (TP+TN)/total_data;
recall = TP/(TP+FN);
precision = TP/total_data;

# TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta=0.5
fscore = (1+beta**2)*precision*recall/((beta**2)*precision+recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# ###  Supervised Learning Models
# **Three models are chosen here**

# ### Question 2 - Model Application
# For each model chosen
# 
# - Describe one real-world application in industry where the model can be applied. 
# - What are the strengths of the model; when does it perform well?
# - What are the weaknesses of the model; when does it perform poorly?
# - What makes this model a good candidate for the problem, given what you know about the data?

# **Answer: **
# 
# **1. Gaussian Naive Bayes (GaussianNB)** 
# 
# - This model is used for finding the spam emails. 
# - It is easy to implement and it is efficent. It could work for many features. 
# - It performs poorly for phrases that encompass multiple words and have distinctive meanings. Besides, this model has a very strong assumption on the shape of data distribution such as every two features are independent.
# - This dataset has 103 features after one-hot encoding which is a lot, besides, we could think of <=50,000 or >50,000  as spam or not-spam. Basically we have 103 features and only two options in outcome as 0 or 1 (<=50,000 or >50,000). So GaussianNB model should be helpful.   
# 
# **2. Decision Trees**
# 
# - The model is used for selecting a flight to travel. 
# - It is easy to use, they are representable graphically and this allows interpretation much easier. It allows having bigger classifiers. They easily handle feature interactions and they’re non-parametric ([References](http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/)). 
# - It is prone to overfitting, specially if there are many features. It is important to control the parameters to avoid overfitting. 
# -  After one-hot encoding, we have 103 features. I am very confident that the decision trees model can handle all features. This model is easy to implement. a table of data and they will build a classifier directly from that data without needing any up front design work to take place ([Reference](https://stackoverflow.com/questions/10317885/decision-tree-vs-naive-bayes-classifier)). 
# 
# **3. AdaBoostClassifier**
# 
# - text classification
# - It could be ued with many different classifieres
# - AdaBoost can be sensitive to noisy data and outliers.
# - We cleaned our data. So, we should not have any outlier or noises that affects the model. Besides, can deal with AdaBoostClassifier can deals with many features. This model can assign different weights to different features based on their importance.Since we have 103 features in the data, AdaBoostClassifier should be a great candidate for machine learning.

# ### Implementation - Creating a Training and Predicting Pipeline
# To properly evaluate the performance of each model you've chosen, it's important to create a training and predicting pipeline that allows to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. The implementation here will be used in the following section.
# In the code block below, I implement the following:
#  - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Fit the learner to the sampled training data and record the training time.
#  - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
#    - Record the total prediction time.
#  - Calculate the accuracy score for both the training subset and testing set.
#  - Calculate the F-score for both the training subset and testing set.
#    - Make sure that you set the `beta` parameter!

# In[32]:


# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score 

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end-start;
        
    # Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end-start;
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    from sklearn.metrics import accuracy_score, fbeta_score
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    # Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# ### Implementation: Initial Model Evaluation
# 
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
#   - Use a `'random_state'` for each model you use, if provided.
# - Calculate the number of records equal to 1%, 10%, and 100% of the training data.
#   - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.

# In[33]:


# Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
#from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(random_state=42)
#clf_C = SVC(random_state=42)
clf_C = AdaBoostClassifier(random_state=42)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = int(len(y_train))
samples_10 = int(0.1*len(y_train))
samples_1 = int(0.01*len(y_train))

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


# ----
# ## Improving Results
# In this final section, I will choose from the three supervised learning models the *best* model to use on the student data. I will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 

# ### Question 3 - Choosing the Best Model
# 
# * Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000. 

# **Answer:**
# Obviously, the training and prediction times of the AdaBoostClassifier are much higher than other two models. However, when 100% of training data is used, both accuracy and F-score values of prediction are larger than the values in GaussianNB and DecisionTreeClassifier. **AdaBoostClassifier** seems to be more suitable compare to other models in fulfilling the goal of project which is prediction of people with \>$50,000 salary.

# ### Question 4 - Describing the Model in Layman's Terms
# 
# * How the final model chosen is supposed to work for CharityML? 

# **Answer:**
# 
# **Intro:** Boosting is an approach to machine learning based on the idea of creating a highly accurate prediction rule by combining many relatively weak and inaccurate rules ([reference](http://rob.schapire.net/papers/explaining-adaboost.pdf)). A weak learner is defined to be a classifier that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification ([Reference](https://en.wikipedia.org/wiki/Boosting_(machine_learning))).
# 
# **Training:** AdaBoostClassifier fits a sequence of weak learners on different weighted training data. It starts by predicting original data set and gives equal weight to each observation. If prediction is incorrect using the first learner, then it gives higher weight to observation which have been predicted incorrectly. Being an iterative process, it continues to add learner(s) until a limit is reached in the number of models or accuracy ([reference](https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/)). Below figure depicts the schematic diagram of AdaBoostClassifier.
# ![image.png](attachment:image.png)
# 
# **Predicting** After training the model, it is time to predict the data! Before prdicting any data, we need to have a strong learner. To convert weak learner to strong learner, we will combine the prediction of each weak learner using methods like ([reference](https://www.analyticsvidhya.com/blog/2015/11/quick-introduction-boosting-algorithms-machine-learning/)):
# - Using average/ weighted average
# - Considering prediction has higher vote
# Then The model is ready to predict the data using the new generated strong learner. 

# ### Implementation: Model Tuning
# - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).

# In[16]:


# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# Initialize the classifier
clf = AdaBoostClassifier(random_state=42)

# Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'n_estimators':[50,200,500,1000],
              'learning_rate':[0.01,0.1,1],
              'algorithm':['SAMME', 'SAMME.R']}

# Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)
                                                                                                                        
# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


# ### Question 5 - Final Model Evaluation
# 
# * What is your optimized model's accuracy and F-score on the testing data? 
# * Are these scores better or worse than the unoptimized model? 
# * How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  

# #### Results:
# 
# |     Metric     | Unoptimized Model | Optimized Model | Naive Predictor  |
# | :------------: | :---------------: | :-------------: | :--------------: |  
# | Accuracy Score |  0.8576           |  0.8676         |  0.2478          | 
# | F-score        |  0.7246           |  0.7448         |  0.2917          | 
# 

# **Answer:**
# 
# The Accuracy and F scores of unoptimized Model	increased from  0.8576 and 0.7246 to 0.8676 and 0.7448 in Optimized Model. These scores in optimized and unoptimized AdaBoostClassifier model are much higher than the Naive Predictor scores which are respectively 0.2478, 0.2917. 

# ----
# ## Feature Importance
# 
# An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.

# ### Question 6 - Extracting Feature Importance
# Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
# 
# In the code cell below, I implement the following:
#  - Import a supervised learning model from sklearn if it is different from the three used earlier.
#  - Train the supervised model on the entire training set.
#  - Extract the feature importances using `'.feature_importances_'`.

# In[17]:


# Import a supervised learning model that has 'feature_importances_'
clf = AdaBoostClassifier(random_state=42)

# Train the supervised model on the training set using .fit(X_train, y_train)
model = clf.fit(X_train, y_train)

# Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)


# ### Feature Selection
# How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model we found earlier, and train it on the same training set *with only the top five important features*. 

# In[18]:


# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


# ### Question 7 - Effects of Feature Selection
# 
# * How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
# * If training time was a factor, would you consider using the reduced data as your training set?

# **Answer:**

# #### Results:
# 
# |     Metric     | Unoptimized Model | Optimized Model | Reduced Features Model  |
# | :------------: | :---------------: | :-------------: | :--------------: |  
# | Accuracy Score |  0.8576           |  0.8676         |  0.8433          | 
# | F-score        |  0.7246           |  0.7448         |  0.7053          | 
# 
# Both Accuracy and F scores are reduced by ~2 and 4\%. The answer of this question depends to the cost of project! If we are under the pressure of time and reducing a couple of percentages of accuracy and F scores of the prediction do not dramatically increase the cost of project, I would suggest using the reduced features. Othewise, I use more features which is giving a better resuts. 
