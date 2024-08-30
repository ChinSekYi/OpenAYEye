import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# import termplotlib as tpl

def plot_results(df, train_len, val_len, test_len):
	plt.figure(figsize=(14,7))
	plt.title(f'Model Performance')
	plt.xlabel('Date', fontsize=18)
	plt.ylabel('Close Price USD($)', fontsize=18)
	plt.plot(df.loc[:, ['Close']], label='Actual Close')

	start_index = 0
	plt.plot(df[['Predictions']].iloc[start_index: start_index + train_len, :], label='Train Predictions')

	start_index += train_len
	plt.plot(df[['Predictions']].iloc[start_index: start_index + val_len, :], label='Validation Predictions')

	start_index += val_len
	plt.plot(df[['Predictions']].iloc[start_index: start_index + test_len, :], label='Test Predictions')

	plt.ylim((0, 1.1*df.loc[:, ['Close']].max(axis=0)[0]))
	
	plt.legend(loc='lower right')
	plt.show()