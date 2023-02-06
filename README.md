# MMJ Analytics
<br />
<br />
<br />



# Lending Factor Analysis and Profit Optimization
![](images/loan_cover_image.jpg)

***
<br />

## Table of Contents

<br />
<br />

***
## **Introduction**
***
The dataset (found [here](https://drive.google.com/file/d/1WFvu8dnVwZV5WuluHFS_eCMJv3qOaXr1/view)) used for this project contains feature data about applicants for personal loans as well as the end result of whether the borrower paid off or defaulted on the loan. The goal of the analysis was to understand the distrbution of the applicant data, analyze the factors that most affect default status, and build a model to predict whether a borrower will default on a loan to optimize lending practice and profit.
<br />
<br />

***
## **Overall**
***
After removing data entries with missing data and extreme outliers, twelve different features were used in the project analysis. One important note about this dataet is that it is imbalanced. The amount of loans that defaulted on payment is significantly smaller than the amount of loans that were paid back, as can be seen in the chart below. Loans that were paid off have been denoted as "Good Loans" while loans that defaulted have been denoted as "Bad Loans."
***
![](images/loan_count_piechart.png)
***
The amount of loans that were paid off outnumber the number of defaults 4:1. This is something that had to be accounted for during analysis and when building the prediction model. It is important to note that the model will be skewed toward predicting that a borrower will not default. Correcting this using methods for an imbalanced dataset tune the model to focus prediction on the default class.
<br />
<br />

***
## **Exploratory Data Analysis**
***
Each individual feature has its own distribution and impact on the end result of whether or not a borrower defaults on their loan. This next section focuses on visualizing each of these distributions to get a full understanding of the dataset.
<br />
<br />

### Numerical Features
***
![](images/numerical_features_eda_1.png)
![](images/numerical_features_eda_2.png)
<br />
<br />

### Categorical Features
***
![](images/categorical_features.png)
<br />
<br />

### Binary Features
***
![](images/binary_features.png)
<br />
<br />

### Annual Income
***
![](images/annual_income.png)
<br />
<br />

It is important to note that during this data visualization process, data cleaning was used to remove extreme outliers that would have an abnormally large impact on the overall analysis. Each distribution of the feature data prviously showed contains data from both good and bad loans in every category.
<br />
<br />

***
## **Hypothesis Testing**
***
After exploring the data, the next step was to use hypothesis testing to answer specific questions about how certain features impact the end result of defaulting on a loan. Since the end result is a predictive model, it is important to first understand if the feature variables have a significant impact on the end result. Using hypothesis testing, it can be determined if certain feature data is worth gathering on an individual applying for a loan. If that feature has a significant impact on the default rate, then it is worth the time to collect this data on individuals since it can grant insight on whether or not they will default on the loan.
<br />
<br />

### Prior Delinquency
***
The first hypothesis test assesed the difference in the default rate based on the binary feature of whether or not the applicant had a prior deliquency, or past unpaid debt. A one-tailed t test with 95% confidence was used to determine if applicants without a prior deliquency had a statstically lower default rate. Based on repeated sampling from applicants with and without a prior deliquency, it was found that applicants without a prior deliquency were statistically less likely to default on loan payments.
![](images/prior_delinq_hypotest.png)
<br />
<br />

### Loan Term Length
***
The second hypothesis test assesed the default rate given different lengths of the term of the loan. This feature was binned into just two categories in the dataset, 36 and 60 months. A one-tailed t test with 95% confidence was used to determine if applicants applying for a shorter term loan have a lower default rate than applicants applying for a longer term loan. The test found that borrowers have a statistically lower default rate with a 36 month loan term than with a 60 month term. This is important because the company can potentially reduce risk by having more short-term loans, or charge a higher interest rate on loans that are longer term.
![](images/loan_term_hypotest.png)
<br />
<br />

### Home Ownership Status
***
The third hypothesis test assesed the default rate given different home ownership status. The categories of owning and mortgaging a home were grouped together to compare to those borrowers who rent a home. A one-tailed t test with 95% confidence was used to determine if applicants who own or mortgage a home have a lower default rate than applicants who rent their home. The test found that borrowers who own or mortgage their home have a statistically lower default rate than those who rent their home.
![](images/home_own_hypotest.png)
<br />
<br />

***
## **Predictive Model**
***
The goal of the predictive model is to be able to maximize overall profit gained from loaning money to the applicants. Each loan that is paid back gives a certain amount of revenue, and each loan given out that is not paid back results in a cost. Arbitrary numbers were used for the sake of building the model, but real numbers could be easily inserted in a real business case. 

The first step in building the model was to create dummy variables for all categorical columns. Once all of these columns ahve a binary value, the train-test split was completed to create the test set and the holdout set. The test set was further split into the training set and the test set against which to test the model before ever seeing the holdout set to prevent data leakage. The training and test set were normalized and all features were included in creating the logistic regression model. 
