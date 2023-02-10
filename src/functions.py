import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import scipy.stats as spicystats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix



def clean_dataset(filepath):
    '''
    Cleans dataset and removes null values and outliers

    Inputs:

    filepath: str with relative filepath to dataset

    Outputs:

    df: cleaned dataset as a dataframe
    '''

    df = pd.read_csv(filepath)

    # 'last_major_derog_none' feature has 97.13% null values, this column is dropped below
    df.drop('last_major_derog_none', axis=1, inplace=True)

    # grade feature is dropped because it is the evaluation of risk on the given loan,
    # presumably using the same data given in the other features to predict loan default

    # ID feature is also dropped, it is just an identifier and not connected to other data
    df.drop('grade', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)

    # all other rows with null values are dropped
    df.dropna(inplace=True)
    # dataset now has 18371 entries instead of 20000

    # 'term' feature had varying punctuation and a added space in the string, replace
    # method was used to standardize the entries
    df.replace({' 36 months': '36 months', ' 36 Months': '36 months', ' 60 months': '60 months'}, inplace=True)

    # 'revol_util' column contained one large outlier that is likely a mis-entry
    # column represents revolving credit utilization rate as a percentage
    # value of 5010.0 is likely a mis-entry as the second highest value is 128.1
    df = df[df['revol_util'] <= 200]

    return df





def sampling_distrbution_hypothesis_test(series1, series2, alpha, sample_size=500, num_samples=500, dist_labels=None):
    
    ''' 
    Takes in 2 numpy series and graphs the distribution of their sample means
    Draws a vertical line at the critical value based on user input alpha value

    Inputs:
    
    series1: (null) series for hypothesis test
    series2: (alt) series for hypothesis test
    alpha: signifiance level of hypothesis test
    sample_size: number of samples
    num_samples: size of sample distribution
    dist_labels: labels for null and alt hypothesis distributions

    Outputs:

    None: graphs the hypothesis test with both sampling distributions
    '''
    
    np.random.seed(seed=8)

    series1_dist = spicystats.norm(np.mean(series1), np.std(series1))
    series2_dist = spicystats.norm(np.mean(series2), np.std(series2))
    
    series1_mean_list = []
    series2_mean_list = []
    
    for i in range(num_samples):
        a = series1_dist.rvs(size=sample_size)
        series1_mean_list.append(np.mean(a))
        b = series2_dist.rvs(size=sample_size)
        series2_mean_list.append(np.mean(b))
    
    series1_sampdist_mean = np.mean(series1_mean_list)
    series1_sampdist_stderror = np.std(series1_mean_list) / np.sqrt(sample_size)
    
    series2_sampdist_mean = np.mean(series2_mean_list)
    series2_sampdist_stderror = np.std(series2_mean_list) / np.sqrt(sample_size)
    
    series1_sampdist = spicystats.norm(series1_sampdist_mean, series1_sampdist_stderror)
    series2_sampdist = spicystats.norm(series2_sampdist_mean, series2_sampdist_stderror)

    series1_xs = np.linspace((series1_sampdist_mean-(4*series1_sampdist_stderror)), \
    (series1_sampdist_mean+(4*series1_sampdist_stderror)), 251)
    series2_xs = np.linspace((series2_sampdist_mean-(4*series2_sampdist_stderror)), \
    (series2_sampdist_mean+(4*series2_sampdist_stderror)), 251)

    fill_space_x1 = series1_xs[series1_xs <= series1_sampdist.ppf(alpha)]

    fig, ax = plt.subplots()
    ax.plot(series1_xs, series1_sampdist.pdf(series1_xs), color='tab:blue', label=dist_labels[0])
    ax.plot(series2_xs, series2_sampdist.pdf(series2_xs), color='tab:orange', label=dist_labels[1])
    ax.axvline(series1_sampdist.ppf(alpha), linestyle='--', color='r', label=f'Critical Value: {(1-alpha)*100}% Confidence')
    ax.axvline(series1_sampdist_mean, linestyle='--', color='tab:blue', label=f'Mean: {series1_sampdist_mean: .3f}')
    ax.axvline(series2_sampdist_mean, linestyle='--', color='tab:orange', label=f'Mean: {series2_sampdist_mean: .3f}')
    ax.get_xticks()
    ax.fill_between(fill_space_x1, 0, series1_sampdist.pdf(fill_space_x1), alpha=0.5, color='b')
    ax.legend()
    ax.set_title('Sampling Distribution Mean of Default Rate')
    ax.set_xlabel('Default Rate')
    ax.set_ylabel('Probability Density')
    ax.set_yticklabels(['', '', '', '', '', '', ''])
    fig.set_size_inches(8, 6)
    fig.tight_layout()

    return None




def best_log_model(X_train, X_test, y_train, y_test, threshold_range, penalty_range, profit_matrix):
    
    '''
    function itereates through different threshold ranges and penalty values
    to find the logistic regression model that maximizes profit

    Inputs:

    X_train: feature data to train the model
    X_test: feature data to test the model
    y_train: target data to train the model
    y_test: target data to evaluate model performance
    threshold_range: list of threshold values, divided by 10 in function
    penalty_range: numpy array of values for Ridge regression penalty
    profit_matrix: numpy array pf profit/loss for each category in predictive matrix

    Outputs:

    max_threshold: threshold value at which profit is maximized
    max_penalty: penalty value at which profit is maximized
    max_profit: maximmum profit produced by the model
    '''
    
    fig, ax = plt.subplots()

    max_threshold = 0
    max_penalty = 0
    profit_matrix = profit_matrix
    max_profit = 0

    for idx, val in enumerate(threshold_range):
        profit_dict = {}
        for j in penalty_range:
            log_model = LogisticRegression(random_state=8, penalty='l2', C=j, max_iter=200).fit(X_train, y_train)
            y_probs = log_model.predict_proba(X_test)
            threshold = (val/10)
            predictions = np.where(y_probs[:,1] >= threshold, 1, 0)
            cm_model = confusion_matrix(y_test, predictions)
            profit_using_model = sum(sum(cm_model * profit_matrix))
            profit_dict[j] = profit_using_model

        max_value = np.max([k for k in profit_dict.values()])
        ax.plot(profit_dict.keys(), profit_dict.values(), label=f'Threshold Value: {val/10} | Max Profit: {max_value: .2f}')

        if profit_using_model > max_profit:
            max_profit = profit_using_model
            max_threshold = val/10
            max_key = [key for key, value in profit_dict.items() if value == max_profit]
            max_penalty = max_key[0]

    ax.legend()
    ax.set_xlabel('Penalty Value')
    ax.set_ylabel('Profit')
    fig.tight_layout()
    fig.set_size_inches(8, 4)

    return max_threshold, max_penalty, max_profit



if __name__ == '__main__':
    pass