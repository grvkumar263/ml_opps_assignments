# PART: library dependencies: -- sklearn, tenserflow, numpy
# import required packages
import matplotlib.pyplot as plt

# import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


GAMMA = 0.001
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


# PART: Define the model
# Create a classifier: a support vector classifier
clf = svm.SVC()

# PART: setting up hyper parameter
hyper_param = {'gamma': GAMMA}
clf.set_params(**hyper_param)

# PART: train model
# Learn the digits on the train subset
clf.fit(X_train, y_train)


# PART: get test set pridection
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
    
# PART: Compute evaluation Matrics 
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
) 