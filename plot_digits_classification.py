# PART: library dependencies: -- sklearn, tenserflow, numpy
# import required packages
import matplotlib.pyplot as plt

# import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. set the range of hyperparameters
gamma_list  = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


# 2. for every combiation of hyper parameter values
hyp_para_combo = [{"gamma":g, "C":c} for g in gamma_list for c in c_list]

assert len(hyp_para_combo) == len(gamma_list)*len(c_list)


# model_hyperparams
# GAMMA = 0.001
# C = 0.5

train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1


# PART: -load dataset --data from files, csv, tsv, json, pickle
digits = datasets.load_digits()


# PART: sanity check visulization of data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)
    

# PART: data pre-processing -- to normlize data, to remove noice,
#                               formate the data tobe consumed by model
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# PART: define train/dev/test splite of experiment protocol
# trin to train model
# dev to set hyperperameter of the model
# test to evaluate the performane of the model

# 80:10:10 train:dev:test

dev_test_frac = 1- train_frac
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size=dev_test_frac, shuffle=True
)


fraction_want = dev_frac/(dev_frac+test_frac)
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=fraction_want, shuffle=True
)
# if testing on the same as traning set: the performance matrics may overestimate 
# the goodness of the model. 
# We want to test on "unseen" sample. 

best_hyp_param = None
best_model = None
accuracy = 0
for curr_param in hyp_para_combo:
    # PART: Define the model
    # Create a classifier: a support vector classifier
    clf = svm.SVC()

    # PART: setting up hyper parameter
    hyper_param = curr_param
    clf.set_params(**hyper_param)

    # PART: train model
    # 2a. train the model
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)


    #Accuracy for the training set.
    curr_predicted_train = clf.predict(X_train)
    curr_accuracy_train = accuracy_score(y_train, curr_predicted_train)    

    #Accuracy for the dev set.
    curr_predicted_dev = clf.predict(X_dev)
    curr_accuracy_dev = accuracy_score(y_dev, curr_predicted_dev)

    # PART: get test set pridection
    # Predict the value of the digit on the test subset
    curr_predicted = clf.predict(X_test)


    # 2b. compute accuracy on validation set
    curr_accuracy = accuracy_score(y_test, curr_predicted)

    #Printing the results for each set of hyperparameters.
    print(f"{hyper_param} \tTest_Accuracy: {curr_accuracy} \tTrain_Accuracy: {curr_accuracy_train} \tDev_Accuracy: {curr_accuracy_dev}")


    # 3. identify best set of hyper parameter for which validation set acuuracy is highest
    if accuracy < curr_accuracy:
        best_hyp_param = hyper_param
        accuracy = curr_accuracy
        accuracy_train = curr_accuracy_train
        accuracy_dev = curr_accuracy_dev
        best_model = clf
        
print("\n")    
print(f"\tBest: {best_hyp_param} \tTest_Accuracy: {accuracy} \tTrain_Accuracy: {accuracy_train} \tDev_Accuracy: {accuracy_dev}")

# if predicted < curr_predicted:
#     predicted = curr_predicted
predicted = best_model.predict(X_test) 

# PART: sanity check visulization of data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, curr_predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
# PART: Compute evaluation Matrics 
# 4. report the best set accuracy with that best model.
"""
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
) 
"""