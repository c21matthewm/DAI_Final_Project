import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import scipy.stats as spicystats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix






def sampling_distrbution_hypothesis_test(series1, series2, alpha, sample_size=500, num_samples=500, dist_labels=None):
    series1_dist = spicystats.norm(np.mean(series1), np.std(series1))
    series2_dist = spicystats.norm(np.mean(series2), np.std(series2))
    
    series1_mean_list = []
    series2_mean_list = []
    
    for i in range(num_samples):
        a = series1_dist.rvs(sample_size)
        series1_mean_list.append(np.mean(a))
        b = series2_dist.rvs(sample_size)
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

    fill_space_x1 = series1_xs[series1_xs <= series1_sampdist.ppf(0.025)]

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

        # max_value = np.max([k for k in f1_dict.values()])
        # max_key = [key for key, value in f1_dict.items() if value == max_value]
        # print(max_key, max_value)
        # if max_value > max_f1:
        #     max_f1 = max_value
        #     max_threshold = val/10
        #     max_penalty = max_key
        # else:
        #     pass

        if profit_using_model > max_profit:
            max_profit = profit_using_model
            max_threshold = val/10
            max_key = [key for key, value in profit_dict.items() if value == max_profit]
            max_penalty = max_key[0]

        # ax.plot(profit_dict.keys(), profit_dict.values(), label=f'Threshold Value: {val/10} | Max F1: {max_profit: .2f}')
    ax.legend()
    ax.set_xlabel('Penalty Value')
    ax.set_ylabel('Profit')
    fig.tight_layout()
    fig.set_size_inches(8, 4)

    return max_threshold, max_penalty, max_profit



if __name__ == '__main__':
    pass