# Machine Learning Engineer Nanodegree
## Capstone Project
Arthur Vetori  
December 31st, 2050

## I. Definition

### Project Overview
With the development of the computer processing power and machine learning techniques, data science become more popular and demanded in the all business sectors. One of the promising application areas is Banking and Financial Industry. The large amount of data that banks have, such as personal education, income and spending habits makes predictive models a strong tool for direct marketing. The wide spread use of internet turns decision processes more based in data than it ever was. In bank marketing, for example, large amount of data with different techniques are used for classifying potential clients for a specific product.

Data scientists can use data points and trends to help build  strategies, tweak content to meet demand, and measure the outcomes of the actions taken by marketers [3].

### Problem Statement
The goal of this project is to classify if the client will subscribe for a term deposit based on multiple client features such as personal information, last contact of the current marketing campaign and economic context attributes. Different techniques will be used and compared in accuracy terms.

Classifier models used in this project will be compared to Decision Trees as benchmark model. The models used are:

- Gaussian Naive-Bayes
- Random Forests
- AdaBoost Classifier
- Support Vector Machines

Feature selection with *SelectKBest* with F-test was used to choose most important features for model tweaking and reduce expensive training time of SVMs.

### Metrics
Metrics used for this this project are: Accuracy, Recall and F1-Score.

Accuracy is the most common metric score for prediction models. Recall is also an important metric for the goal of this project because it measures the false negatives (predicted as the client will not subscribe for the term deposit when it possibly could subscribe) that are predicted by the model, meaning that the bank are leaving money on the table. F1-score is the balance between Recall and Precision, in other words, the balance between false negatives and false positives.

Those metrics are defined as:

- Accuracy = (TP + TN)/(TP + TN + FP + FN)
- Precision = TP/(TP + FP)
- Recall = TP/(TP + FN)
- F1-Score = (Precision * Recall) / (Precision + Recall)

Where:
- TP = True positives
- TN = True negatives
- FP = False positives
- FN = False negatives




## II. Analysis

### Data Exploration

The data is related with direct marketing campaigns of a Portuguese banking institution and contains features related to client data, last contact, economic attributes and a label 'y' (output) with values 'yes' or 'no' for the term deposit subscription. The marketing campaigns were based on phone calls.

The dataset used in this project is available at:

> https://archive.ics.uci.edu/ml/datasets/bank+marketing

The file downloaded from the link contains two datasets, one with 20 features (__bank-additional-full.csv__) and another with 17 features (__bank-full.csv__). The dataset used in this project is __bank-additional-full.csv__. This dataset, consists of 41188 registers and are ordered by date (from May 2008 to November 2010). There is another dataset named __bank-additional.csv__ with 10% of data points randomly selected from __bank-additional-full.csv__, that will be used in the final model.

Inputs are:

* __age__ (numeric)  

* __job__ : type of job (categorical:   'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')  

* __marital__ : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)  

* __education__ : (categorical:'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course',
'university.degree','unknown')  

* __default__: has credit in default? (categorical: 'no','yes','unknown')  

* __housing__: has housing loan? (categorical: 'no','yes','unknown')  

* __loan__: has personal loan? (categorical: 'no','yes','unknown')  

* __contact__: contact communication type (categorical: 'cellular','telephone')

* __month__: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')

* __day_of_week__: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')

* __duration__: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

* __campaign__: number of contacts performed during this campaign and for this client (numeric, includes last contact)

* __pdays__: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)

* __previous__: number of contacts performed before this campaign and for this client (numeric)

* __poutcome__: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

* __emp.var.rate__: employment variation rate - quarterly indicator (numeric)

* __cons.price.idx__: consumer price index - monthly indicator (numeric)

* __cons.conf.idx__: consumer confidence index - monthly indicator (numeric)

* __euribor3m__: euribor 3 month rate - daily indicator (numeric)

* __nr.employed__: number of employees - quarterly indicator (numeric)

Sample of the data (some columns were hidden for presentation purposes):


| age | job | marital | education | default | ... | cons.conf.idx | euribor3m | nr.employed | y |
|-----|-----------|---------|-------------|---------|-----|---------------|-----------|-------------|----|
| 56 | housemaid | married | basic.4y | no | ... | -36.4 | 4.857 | 5191.0 | no |
| 57 | services | married | high.school | unknown | ... | -36.4 | 4.857 | 5191.0 | no |
| 37 | services | married | high.school | no | ... | -36.4 | 4.857 | 5191.0 | no |
| 40 | admin. | married | basic.6y | no | ... | -36.4 | 4.857 | 5191.0 | no |
| 56 | services | married | high.school | no | ... | -36.4 | 4.857 | 5191.0 | no |

Statistics of numerical features (employement indexes were removed).

|  | age | duration | campaign | pdays | previous | cons.price.idx | cons.conf.idx | euribor3m |
|-------|-------------|--------------|--------------|--------------|--------------|----------------|---------------|--------------|
| count | 41188.00000 | 41188.000000 | 41188.000000 | 41188.000000 | 41188.000000 | 41188.000000 | 41188.000000 | 41188.000000 |
| mean | 40.02406 | 258.285010 | 2.567593 | 962.475454 | 0.172963 | 93.575664 | -40.502600 | 3.621291 |
| std | 10.42125 | 259.279249 | 2.770014 | 186.910907 | 0.494901 | 0.578840 | 4.628198 | 1.734447 |
| min | 17.00000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 92.201000 | -50.800000 | 0.634000 |
| 25% | 32.00000 | 102.000000 | 1.000000 | 999.000000 | 0.000000 | 93.075000 | -42.700000 | 1.344000 |
| 50% | 38.00000 | 180.000000 | 2.000000 | 999.000000 | 0.000000 | 93.749000 | -41.800000 | 4.857000 |
| 75% | 47.00000 | 319.000000 | 3.000000 | 999.000000 | 0.000000 | 93.994000 | -36.400000 | 4.961000 |
| max | 98.00000 | 4918.000000 | 56.000000 | 999.000000 | 7.000000 | 94.767000 | -26.900000 | 5.045000 |

Categorical features

|  | job | marital | education | default | housing | loan | contact | month | day-of-week | poutcome | y |
|--------|--------|---------|-------------------|---------|---------|-------|----------|-------|-------------|-------------|-------|
| count | 41188 | 41188 | 41188 | 41188 | 41188 | 41188 | 41188 | 41188 | 41188 | 41188 | 41188 |
| unique | 12 | 4 | 8 | 3 | 3 | 3 | 2 | 10 | 5 | 3 | 2 |
| top | admin. | married | university.degree | no | yes | no | cellular | may | thu | nonexistent | no |
| freq | 10422 | 24928 | 12168 | 32588 | 21576 | 33950 | 26144 | 13769 | 8623 | 35563 | 36548 |

Approximately only 12% of data points are flagged as 'yes' in the output variable and 60% of all contacts were made in May thus the dataset is imbalanced. There are many values marked as 'unknown' or 'nonexistent'


### Exploratory Visualization

#### Univariate Analysis

![](img/histogram_1.png)
![](img/histogram_2.png)
![](img/histogram_3.png)
![](img/histogram_4.png)

![](img/count_1.png)
![](img/count_2.png)
![](img/count_3.png)
![](img/count_4.png)

- More than half of the clients in the dataset are married
- More than 75% of the clients did not previously defaulted in credit
- Almost half of the clients in the dataset does not have housing loan. This could be a good indicator for the subscription of time deposit, since it may indicate that the client maybe have money savings
- Aproximately 85% of the clients doesn't have loans. Probably the calls are made to clients who do not have loans.
- Aproximately 60% of the jobs in the dataset are composed from administrators, blue-collars and technicians
- Aproximately 30% of the clients have university degree
- Most of the calls were made in may
- Aproximately 12% of the clients subscribed for the term deposit

#### Multivariate Analysis

![](img/cat_1.png)![](img/cat_2.png)
![](img/cat_3.png)![](img/cat_4.png)
![](img/cat_5.png)![](img/cat_6.png)

Facet Histogram

![](img/facet_1.png)
![](img/facet_2.png)

![](img/scatter_1.png)![](img/scatter_2.png)
![](img/scatter_3.png)![](img/scatter_8.png)

![](img/scatter_4.png)
![](img/scatter_5.png)
![](img/scatter_6.png)
![](img/scatter_7.png)


Correlation heatmap of numerical features
![](img/heat_map.png)

- Most of the clients that subscribed for the term deposit does not have loans
- Most of the clients that subscribed for the term deposit have university degree
- Last call duration seems to be longer for married clients
- Calls made in May shows a low conversion rate (subscriptions/calls).
- Up to 10 contacts in the campaign seems to be more successful for client subscriptions.
- Number of contacts to the same client seems to be higher when the euribor is higher. This due to the fact that term deposit are offered with higher fixed rate when the risk-free interest rate is higher, so it is more probable that client will subscribe for the term deposit.
- Euribor is higher when consumer price index is higher, as we can see in the heatmap. Central bank manipulate short-term interest rates to maintain inflation around a certain target rate. Higher inflation levels leads to high interest rate levels.
- Euribor displays high positive correlation with employment variation rate.


### Algorithms and Techniques

For this project, the follower classifiers were used for prediction

- __Gaussian Naive-Bayes__: Naive-Bayes is a classifier which uses the Bayes Theorem, predicting membership probabilities for each class such as the probability that given data point belongs to a specific class. Relatively simple to understand and implement, very fast training/testing times, works very well with small sized datasets and it isn't sensitive to irrelevant features. However, it assume that every feature is independent of each other, which it isn't always the case.

- __Random Forests__: Random Forests is an ensemble method that uses Decision Trees as a weak-learner, building multiple trees and merging them together to achieve more accuracte and stable predictions. The algorithm is parallelizable and it is less likely to overfitting than Decision Trees. Works very well with imbalanced datasets.

- __AdaBoost Classifier__: Ensemble method that combine weak-learners to form a strong-rule. This is done by training the dataset with a weak-learner and adjust feature weights to compose a final strong classifier. Some of the advantages of using AdaBoost is it fast training/testing time, simple to implement and it is versatile. There are drawkbacks such as weak classifiers that overfit and it vulnerability to uniform noise.

- __SVMs__: Based on the idea of finding a hyperplane that best divide a dataset into two classes, its a powerful algorithm that works very well in smaller datasets. The algorithm is weak when the dataset display multiple overlapping data points with different classes and it is training/testing time is very expensive.

- __Decision Trees__: Works building 'trees' with a 'if-else' decision rule, splitting data into subsets on a specific variable. It is easy to interpret results, works well with noisy data and is very versatile. Tends to overfit and is unstable.

To avoid overfitting and achieve more realistic predictions, cross-validation was used to divide the dataset in training and testing samples.
Grid Search was used to hyperparameter tunning that will be discussed in the __Refinement__ section.


### Benchmark
Decision Trees has been chosen as the Benchmark model due to it vesatility and historical commonly application in the business industry. It was compared with all other algorithms discussed in the previous section. GridSearch also was applied to Decision Trees to obtain the best hyperparameters.


## III. Methodology

### Data Preprocessing

Some data cleaning was done after exploration of the data.

- Data points containing 'unknown' values in 'marital', 'default', 'housing', 'loan', 'job', 'education'.
- 'poutcome' was removed because most of the values were labeled as 'nonexistent'
- 'pdays' was removed because it is related to days passed after the last contact and most of the clients weren't contacted before the campaign.
- 'emp.var.rate', 'cons.price.idx' and 'nr.employed' were removed because they are economy indexes that are highly correlated with each other and with 'euribor3m'.
- 'duration' was dropped for more realistic prediction purposes. As described in the dataset documentation, this variable highly affects the output, for duration 0 the output variable will be 'no'.
- Categorical features with 'yes' and 'no' values were mapped into numerical 1 and 0.
- Categorical features with multiple levels were converted into dummies using pandas get_dummies() method.

After the data cleaning, feature selection was done using scikit-learn SelectKBest with F-test to retrieve most important features.

SelectKBest scores:


| feature | score |
|----------------------------------|-----------------------|
| 'euribor3m', | 3378.7979831964785 |
| 'previous', | 1671.7384183749552 |
| 'month_mar', | 666.76307496622996 |
| 'contact', | 644.25062942995714 |
| 'month_oct', | 621.59538809456456 |
| 'month_sep', | 479.02158927356442 |
| 'month_may', | 391.23382771314948 |
| 'job_retired', | 322.90632930416552 |
| 'job_student', | 241.30200550373138 |
| 'month_apr', | 185.476211929545 |
| 'month_dec', | 170.61102112665378 |
| 'campaign', | 146.71288654748687 |
| 'job_blue-collar', | 139.52127387424932 |
| 'cons.conf.idx', | 116.26589635142068 |
| 'age', | 72.55591257547961 |
| 'education_university.degree', | 66.401032282512446 |
| 'education_basic.9y', | 64.092303503479798 |
| 'marital_single', | 54.364050294515025 |
| 'job_services', | 37.555115181004162 |
| 'month_jul', | 36.771991523764861 |
| 'marital_married', | 30.32781607581321 |
| 'job_admin.', | 17.606245555463541 |
| 'month_nov', | 17.563914596321414 |
| 'day_of_week_mon', | 14.298991290761432 |
| 'job_unemployed', | 13.344229174501237 |
| 'job_entrepreneur', | 11.693984248874791 |
| 'education_basic.6y', | 10.817494308733895 |
| 'day_of_week_thu', | 8.6616278065727261 |
| 'month_aug', | 7.8202103783246173 |
| 'job_technician', | 5.3933656447355043 |
| 'day_of_week_fri', | 4.8146115553919442 |
| 'marital_divorced', | 4.546462943348514 |
| 'housing', | 3.0921666580915823 |
| 'education_high.school', | 2.57771535650275 |
| 'education_basic.4y', | 2.5258744429942777 |
| 'day_of_week_tue', | 2.3455018904979035 |
| 'education_illiterate', | 2.1261515152038206 |
| 'day_of_week_wed', | 2.1028214576626532 |
| 'loan', | 0.77017602643879501 |
| 'default', | 0.43477204813567444 |
| 'job_self-employed', | 0.33227395718531533 |
| 'education_professional.course', | 0.19439973308938621 |
| 'job_management', | 0.17966110137136229 |
| 'month_jun', | 0.084019965067283292 |
| 'job_housemaid', | 0.0015171245995779549 |

Features were selected using a threshold of scores above 10.0. New dataset was constructed using
__['euribor3m', 'previous', 'month_mar', 'contact', 'month_oct', 'month_sep', 'month_may', 'job_retired', 'job_student', 'month_apr', 'month_dec', 'campaign', 'job_blue-collar', 'cons.conf.idx', 'age', 'education_university.degree', 'education_basic.9y', 'marital_single', 'job_services', 'month_jul', 'marital_married', 'job_admin.', 'month_nov', 'day_of_week_mon', 'job_unemployed', 'job_entrepreneur', 'education_basic.6y']__
as features.

With the new dataset constructed after the datacleaning and the feature selection, MinMaxScaler was applied to the data to make algorithms more stable. This approach of normalization was because the data is not normally distributed and the there are no problematic outliers in the dataset.

### Implementation

In the implementation process of this work, dependencies were loaded when they needed. The process developed in this work follow as:

1. Loading data using *pandas* framework.
2. Analyzing data using *matplotlib* and *seaborn* statistical data visualization.
3. Data Cleaning
4. Feature selection with *scikit-learn* class *SelectKBest* to select most important features and subset new dataset.
5. Applying *MinMaxScaler* class from *scikit-learn* to the new dataset for normalization.
6. Using *train_test_split* from *scikit-learn* to divide the new dataset into 75% training sample and 25% in testing sample.
7. Defined a score function that returns Accuracy, Precision, Recall, F1-score and F2-score
8. Defined
9. Using *GridSearchCV* from *scikit-learn* to tune hyperparameters. The grid was applied to Decision Trees, AdaBoost, Random Forests and SVM classifier algorithms.
10. Trained and tested all classifier using the best hyperparameters. Time was measured in each training and test process.
11. Compare algorithms based on scores calculated by the *score* function defined before.
12. Applied final algorithm to data contained in *bank-additional.csv* file.
13. Final score output.

### Refinement

Initally, models were trained using default parameters, with exception of SVM.
```
clf = GaussianNB()
...
clf = RandomForestClassifier(random_state=42)
...
clf = AdaBoostClassifier(random_state=42)
...
clf = SVC(C=100, kernel='rbf', gamma = 0.1, random_state=42)
...
clf = DecisionTreeClassifier()
```

The following results were obtained:


| classifier | training_time | testing_time | accuracy | precision | recall | f1_score | f2_score |
|----------------|---------------|--------------|----------|-----------|----------|----------|----------|
| GaussianNB | 0.0156 | 0.0156 | 0.841249 | 0.385333 | 0.428769 | 0.405892 | 0.419316 |
| Random Forest | 0.0624 | 0.0312 | 0.869938 | 0.472109 | 0.239972 | 0.318203 | 0.266145 |
| AdaBoost | 0.187201 | 0.124801 | 0.876629 | 0.539313 | 0.168396 | 0.256653 | 0.195253 |
| SVM | 3.19802 | 1.747211 | 0.875361 | 0.517101 | 0.219571 | 0.308252 | 0.248124 |
| Decision Trees | 0.0156 | 0 | 0.823493 | 0.317602 | 0.344398 | 0.330458 | 0.338683 |

AdaBoost outperformed others classifiers in Accuracy and Precision. GaussianNB performed better in Recall and F1-Score. Given those results, GridSearchCV were applied for hyperparameter tunning with exception of GaussianNB, since prior probabilities are calculated automatically.

Grid parameters supplied for the models:

- Random Forests

```
rf_params = {
                'n_estimators': [200, 500],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [4,5,6,7,8],
                'criterion' :['gini', 'entropy']
            }
```


- AdaBoost

```
ada_params = {
                'n_estimators':[75,200,500],
                'learning_rate':[1.0,1.5,2.0]
             }
```

- SVMs

```
svc_params = {
              'kernel' : ['linear', 'rbf', 'sigmoid'],
              'gamma' : [0.1, 0.01, 0.001, 0.0001],
              'C' : [100, 1000, 10000]
             }
```

- Decision Trees

```
dt_params = {
             'criterion' : ['gini','entropy'],
             'splitter' : ['best', 'random']
            }
```

Best parameters output:

```
Random Forest Classifier best params:{'criterion': 'gini', 'max_depth': 8, 'max_features': 'auto', 'n_estimators': 500}

AdaBoost Classifier best params:{'learning_rate': 1.0, 'n_estimators': 75}

SVC best params: {'C': 10000, 'gamma': 0.01, 'kernel': 'rbf'}

Decision Tree best params: {'criterion': 'entropy', 'splitter': 'random'}

```

Scores using best params:

| classifier | training_time | testing_time | accuracy | precision | recall | f1_score | f2_score |
|----------------|---------------|--------------|----------|-----------|----------|----------|----------|
| GaussianNB | 0.0156 | 0 | 0.840453 | 0.359909 | 0.427027 | 0.390606 | 0.411673 |
| Random Forest | 1.62241 | 0.140401 | 0.892233 | 0.669725 | 0.197297 | 0.304802 | 0.229704 |
| AdaBoost | 0.312002 | 0.0312 | 0.883495 | 0.536232 | 0.2 | 0.291339 | 0.228677 |
| SVM | 26.052167 | 0.218401 | 0.88835 | 0.581699 | 0.240541 | 0.340344 | 0.272505 |
| Decision Trees | 0.0156 | 0 | 0.851456 | 0.389578 | 0.424324 | 0.40621 | 0.416888 |

That showed that GridSearch improved some metrics in the most of the algorithms.

## IV. Results

### Model Evaluation and Validation

Model performance

![](img/result_1.png)
![](img/result_2.png)
![](img/result_3.png)
![](img/result_4.png)

GaussianNB performed better in F1-Score and Recall even after the hyperparameter tunning. Thus, GaussianNB was chosen as the final model for this work to be compared to the Decision Trees benchmark model.

### Justification

As discussed in the previous section and in the benchmark section, based on the recall and f1 scores, GaussianNB performed better in the model training phase.

GaussianNB trained on 75% of the full sample and tested against the 10% randomly selected sample provided in the *bank-additional.csv* file displays a result of:

| Metric | Score |
|----------|--------------------|
| Accuracy | 0.8404 |
| Recall | 0.4270 |
| F1-Score | 0.3906 |

Decision trees splitter was parametrized as 'random', giving different scores whenver it was trained, so the metrics below are output as a mean result of 100 runs.
The Decision Trees benchmark model performed better than the GaussianNB in the mean.

| Metric | Score |
|----------|---------------------|
| Accuracy | 0.8570 |
| Recall | 0.4409 |
| F1-Score | 0.4248 |


## V. Conclusion

### Free-Form Visualization

The final plot below summarizes how the different algorithms performed in the training phase of this project. Even though GaussianNB showed us a better performance in that phase, the final performance of the Decision Trees in the final test were better on average.

![](img/final_1.png)



### Reflection

The nature of the problem demands more robust score methods than accuracy. False negative means that we are leaving money on the table. In this case recall is a more important metric to stick on.

I feel that the dataset used in this project lack of some features that could improve scores, such as number of children, income, gender. Another issue observed, is that the data is imbalanced. Approximately only 12% subscribed for the term deposit and most of contacts were made in may, given that, its more appropriate to not consider seasonality in the prediction model. Top 5 non-multilevel features obtained through feature selection with SelectKBest are: **euribor3m**, **previous**, **contact**, **campaign**, **cons.conf.idx**. Based on those features, we could suggest a better marketing strategy, for example:

**euribor3m**: With low risk-free interest rate, term deposits are offered with a lower interest rate. The bank could offer another type of products that are more attractive in this economic scenario.<br>
**previous**: This is a important feature when looking in the long run. Making more frequent calls to keep the relationship with the client closer will make a huge impact in future possible subscriptions.<br>
**contact**: The dataset showed that contacts through cellular are more efficient than telephone. <br>
**campaign**: This is the same case as **previous** feature. Create a better relationship with the client. <br>
**cons.conf.idx**: When the consumer confidence index is lower, clients will tend to spend less and save more. Usually they will prefer investments with less risk.

### Improvement

After a deep reflection, there are many improvements that could be done to achieve better results in this work, such as feature engineering for example, getting features and grouping in clusters for classifying them based on each cluster that they belong. Many other techniques could be used to chose features more wisely.

-----------

### References
[1] https://archive.ics.uci.edu/ml/datasets/bank+marketing

[2] https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052

[3] https://www.forbes.com/sites/steveolenski/2018/03/06/data-science-is-the-key-to-marketing-roi-heres-how-to-nail-it/#256e4b9231c3

[4] https://www.kaggle.com/janiobachmann/marketing-in-banking-opening-term-deposits
