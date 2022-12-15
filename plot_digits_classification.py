# PART: library dependencies: -- sklearn, tenserflow, numpy
# import required packages
import matplotlib.pyplot as plt
import random
import numpy as np
# import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import preprocess_digits, data_viz, train_dev_test_split, get_all_h_params_comb, train_save_model, get_all_tree_param_comb, tree_h_param_tuning
from joblib import dump,load

# 1. set the range of hyperparameters
gamma_list  = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


tree_depth = [ 5, 16, 7, 18, 9]
min_weight_fraction_leaf = [ 0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

params = {}
params['gamma'] = gamma_list
params['C'] = c_list


tree_param = {}
tree_param['max_depth'] = tree_depth
tree_param['min_weight_fraction_leaf'] = min_weight_fraction_leaf


# 2. for every combiation of hyper parameter values
h_param_comb = get_all_h_params_comb(params)
tree_parm_comb = get_all_tree_param_comb(tree_param)

assert len(h_param_comb) == len(gamma_list)*len(c_list)

def h_param():
    return h_param_comb



train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1


# PART: -load dataset --data from files, csv, tsv, json, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)

# housekeeping
del digits



# PART: define train/dev/test splite of experiment protocol
# trin to train model
# dev to set hyperperameter of the model
# test to evaluate the performane of the model

# 80:10:10 train:dev:test


# if testing on the same as traning set: the performance matrics may overestimate 
# the goodness of the model. 
# We want to test on "unseen" sample. 

# X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
#     data, label, train_frac, dev_frac, 1 - (train_frac + dev_frac), random_state = 5
# )

# X_train, y_train, X_test, y_test = train_test_split(
#     data, label, test_size=0.2, random_state = 5
# )


svm_acc = []
tree_acc = []

svm_complexity = []
tree_complexity = []

for i in range(5):

    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac, 1 - (train_frac + dev_frac)
    )

    model_path, clf, svm_best_hyp_param, svm_run_time = train_save_model(X_train, y_train, X_dev, y_dev, None, h_param_comb)
    
    # tree_clf = tree.DecisionTreeClassifier()
    # tree_clf = tree_clf.fit(X_train, y_train)

    # tree_pre = tree_clf.predict(X_test)

    tree_best_model, tree_accuracy, tree_best_hyp_param, tree_run_time = tree_h_param_tuning(tree_parm_comb, X_train, y_train, X_dev, y_dev)




    best_model = load(model_path)


    predicted= best_model.predict(X_test) 
    tree_predict = tree_best_model.predict(X_test)

    # PART: sanity check visulization of data
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    svm_acc.append(accuracy_score(y_test, predicted))
    tree_acc.append(accuracy_score(y_test, tree_predict))

    svm_complexity.append(svm_run_time)
    tree_complexity.append(tree_run_time)

    print()
    print(f"svm[{i+1}] best hyper param: {svm_best_hyp_param}")
    print(f"tree[{i+1}] best hyper param: {tree_best_hyp_param}")


# PART: Compute evaluation Matrics 
# 4. report the best set accuracy with that best model.
print()
print(f'SVM: {svm_acc}')
print(f'TREE: {tree_acc}')
print()

svm_acc = np.array(svm_acc)
tree_acc = np.array(tree_acc)

svm_mean = np.mean(svm_acc)
tree_mean = np.mean(tree_acc)

svm_variance = np.var(svm_acc)
tree_variance = np.var(tree_acc)

print(f"SVM Mean: {svm_mean} \t Variance: {svm_variance}")
print(f'TREE Mean: {tree_mean} \t Variance: {tree_variance}')
print()


if svm_mean > tree_mean:
    print("svm is best for mean")
    # print(
    #     f"Classification report for classifier {clf}:\n"
    #     f"{metrics.classification_report(y_test, predicted)}\n"
    # ) 

    # print(f"Best hyperparameters were: {best_model}")

else:
    print("tree is best from mean")
print()

print("Other way to decide the model")
print(f"Complexity of SVM: {svm_complexity}")
print(f"Complexity of TREE: {tree_complexity}")

print()