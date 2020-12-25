# Import Dataset and read
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random


# We define a function to compute the power of a list.

def calc_power(list_val, power):
    out = []
    for i in range(len(list_val)):
        out.append(pow(list_val[i], power))
    return out



# We then define a function to determine the model parameters a, b, c of the quadratic equation so that we can predict the output y for any given input x.

def build_Model(x, y):
    model_params = []
    n = len(x)

    X = np.array([[n, sum(x), sum(calc_power(x, 2))],
                  [sum(x), sum(calc_power(x, 2)), sum(calc_power(x, 3))],
                  [sum(calc_power(x, 2)), sum(calc_power(x, 3)), sum(calc_power(x, 4))]])
    xy = [np.dot(x, y)]
    x2y = [np.dot(calc_power(x, 2), y)]
    Y = np.array([[(sum(y))], [(sum(xy))], [(sum(x2y))]])

    model_params = np.dot(np.linalg.inv(X), Y)

    return model_params



# We define a function to predict the output y using the model parameters and the input data x.

def predictOutput(x, y, A1):
    y_predict = A1[2] * calc_power(x, 2) + A1[1] * x + A1[0]
    return y_predict


# RANSAC algorithm:
#     1. Select three data points randomly from the dataset.
#     2. Determine the model parameters for the quadratic from those three data points.
#     3. Compare all the datapoints with the predicted model equation and classify them as inliers or outliers.
#     4. Select a model that maximizes the ratio of inliers to outliers.
#     5. Generate a curve fit from the final model.

def ransac(x_data, y_data, n, t, success_threshold,dataset_id):
    final_inliers_x = []
    final_inliers_y = []
    final_outliers_x = []
    final_outliers_y = []
    worstfit = 0
    prev_inliers = 0
    # Number of iterations
    # n=10000

    # Threshold value
    # t=55

    # Worst possible error is infinite error
    worst_error = np.inf

    for i in range(n):

        dataPoints = random.sample(range(len(x_data)), 3)
        # print(dataPoints)
        possible_inliers_x = []
        possible_inliers_y = []

        for i in dataPoints:
            possible_inliers_x.append(x_data[i])
            possible_inliers_y.append(y_data[i])
        test_Model = build_Model(possible_inliers_x, possible_inliers_y)
        y_predict = predictOutput(x_data, y_data, test_Model)
        # print(possible_inliers_x)
        # print(possible_inliers_y)

        num_inliers = 0
        num_outliers = 0
        valid_inliers_x = [0]
        valid_inliers_y = [0]
        valid_outliers_x = [0]
        valid_outliers_y = [0]

        for i in range(len(x_data)):

            if abs(y_data[i] - y_predict[i]) < t:
                valid_inliers_x.append(x_data[i])
                valid_inliers_y.append(y_data[i])
                num_inliers += 1
            else:
                valid_outliers_x.append(x_data[i])
                valid_outliers_y.append(y_data[i])
                num_outliers += 1

        if num_inliers > worstfit:
            worstfit = num_inliers

            # print("###############################    Better Model Found     #####################################")
            # Update chosen starting points

            input_points_x = possible_inliers_x
            input_points_y = possible_inliers_y

            # Update the model parameters
            update_model = build_Model(valid_inliers_x, valid_inliers_y)
            op = predictOutput(valid_inliers_x, valid_inliers_y, update_model)
            final_model = update_model

            # Update temperary variables to preserve data corresponding to the final chosen model.

            fin_inlier = num_inliers
            fin_outlier = num_outliers
            final_inliers_x = valid_inliers_x.copy()
            final_inliers_y = valid_inliers_y.copy()
            final_outliers_x = valid_outliers_x.copy()
            final_outliers_y = valid_outliers_y.copy()

            success_rate = (worstfit / len(x_data)) * 100

            if success_rate >= success_threshold:
                break
            # print(num_inliers, num_outliers)

    # print(fin_inlier, fin_outlier)
    #
    # print('Worstfit=', worstfit)

    fig = plt.figure(figsize=(10, 5))
    (ax1, ax2) = fig.subplots(1, 2)
    fig.suptitle('RANSAC Output Curve Fit')

    ax1.plot(x_data, predictOutput(x_data, y_data, final_model), color='red', label='Curve Fit')
    ax1.plot(x_data, y_data, 'o', color='blue', label='Input Data')
    ax1.set(xlabel='x-axis', ylabel='y-axis', title="Dataset "+str(dataset_id))
    ax1.legend()

    ax2.plot(x_data, predictOutput(x_data, y_data, final_model), color='red', label='Curve Fit')
    ax2.plot(final_inliers_x, final_inliers_y, 'o', color='black', label='Inliers')
    ax2.plot(final_outliers_x, final_outliers_y, 'o', color='orange', label='Outliers')
    ax2.plot(input_points_x, input_points_y, 'o', color='lime', label='Picked Points')
    ax2.set(xlabel='x-axis', ylabel='y-axis', title="Dataset "+str(dataset_id) )
    ax2.legend()
    # plt.show()




