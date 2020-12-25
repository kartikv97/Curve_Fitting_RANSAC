
# Import Dataset and read
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import utils

data_1 = pd.read_csv(r"..\dataset\data_1.csv")
data_2 = pd.read_csv(r"..\dataset\data_2.csv")


# We convert the dataset into a list and then split the data into separate x and y variable lists.
data_1 = data_1.values.tolist()
data_2 = data_2.values.tolist()
# Split Data
x_data1 = []
y_data1 = []
x_data2 = []
y_data2 = []
for i in range(len(data_1)):
    x_data1.append(data_1[i][0])
    y_data1.append(data_1[i][1])
for i in range(len(data_2)):
    x_data2.append(data_2[i][0])
    y_data2.append(data_2[i][1])

print("Processing Output Plots...")

# We plot both the datasets using functions from the matplotlib library.
fig = plt.figure(figsize=(10,5))
(ax1, ax2) = fig.subplots(1, 2)
fig.suptitle('Original Datasets')
ax1.plot(x_data1, y_data1, 'ro', label='dataset1')
ax1.set(xlabel='x-axis', ylabel='y-axis', title="Dataset 1")
ax1.legend()
ax2.plot(x_data2, y_data2, 'bo', label='dataset2')
ax2.set(xlabel='x-axis', ylabel='y-axis', title="Dataset 2")
ax2.legend()
# plt.show()


# Plot the output curve fit using the LMSE method.
fig = plt.figure(figsize=(10,5))
(ax1, ax2) = fig.subplots(1, 2)
fig.suptitle('Least Mean Square Output Curve Fit')

ax1.plot(x_data1, y_data1, 'bo', label=' Dataset 1')
ax1.plot(x_data1, utils.predictOutput(x_data1, y_data1, utils.build_Model(x_data1, y_data1)), color='red', label='Curve Fit')
ax1.set(xlabel='x-axis', ylabel='y-axis', title="Dataset 1")
ax1.legend()

ax2.plot(x_data2, y_data2, 'o', label=' Dataset 2')
ax2.plot(x_data2, utils.predictOutput(x_data2, y_data2, utils.build_Model(x_data2, y_data2)), color='red', label='Curve Fit')
ax2.set(xlabel='x-axis', ylabel='y-axis', title="Dataset 2")
ax2.legend()
# plt.show()

# We plot the output curve fit for the first dataset using the RANSAC algorithm.
utils.ransac(x_data1, y_data1, 10000, 45, 95,1)


# We plot the output curve fit for the second dataset using the RANSAC algorithm.
utils.ransac(x_data2, y_data2, 10000, 45, 95,2)

print("Finished Processing Generating Output...")
plt.show()