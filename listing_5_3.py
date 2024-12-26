import pandas as pd    
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

min_range = -5 
max_range = 5
xx, yy = np.meshgrid(np.linspace(min_range, max_range, 1000), 
                     np.linspace(min_range, max_range, 1000))

x_data = np.random.normal(loc=2.0, scale=0.5, size=100) 
y_data = np.random.normal(loc=2.0, scale=0.5, size=100)
cluster_1_df = pd.DataFrame(({'A': x_data, 'B': y_data}))
x_data = np.random.normal(loc=-2.0, scale=0.5, size=100)
y_data = np.random.normal(loc=-2.0, scale=0.5, size=100)
cluster_2_df = pd.DataFrame(({'A': x_data, 'B': y_data}))
X_train = pd.concat([cluster_1_df, cluster_2_df]).values

x_data = np.random.normal(loc=2.0, scale=0.4, size=20) 
y_data = np.random.normal(loc=2.0, scale=0.4, size=20)
X_test_normal = pd.DataFrame(({'A': x_data, 'B': y_data})).values

x_data = np.random.normal(loc=0.0, scale=4.0, size=20) 
y_data = np.random.normal(loc=0.0, scale=4.0, size=20)
X_test_outliers = pd.DataFrame(({'A': x_data, 'B': y_data})).values

clf = svm.OneClassSVM(nu=0.1) 
clf.fit(X_train)

y_pred_train     = clf.predict(X_train) 
y_pred_normal    = clf.predict(X_test_normal)
y_pred_outliers  = clf.predict(X_test_outliers)
n_error_train    = y_pred_train.tolist().count(-1)
n_error_normal   = y_pred_normal.tolist().count(-1)
n_error_outliers = y_pred_outliers.tolist().count(-1)

dec_func = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
dec_func = dec_func.reshape(xx.shape)

plt.title("OCSVM") 
plt.contourf(xx, yy, dec_func, 
             levels=np.linspace(dec_func.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, dec_func, levels=[0], linewidths=2, colors="red")
plt.contourf(xx, yy, dec_func, levels=[0, dec_func.max()], colors="green")

train_points = plt.scatter(X_train[:, 0], X_train[:, 1], 
                           c="grey", s=40, marker='.')
normal_points = plt.scatter(X_test_normal[:, 0], X_test_normal[:, 1], 
                            c="blue", s=50, marker="*")
outlier_points = plt.scatter(X_test_outliers[:, 0], X_test_outliers[:, 1], 
                             c="red", s=50, marker='P')

plt.axis("tight")
plt.xlim((min_range, max_range))
plt.ylim((min_range, max_range))
plt.legend(
    [a.collections[0], train_points, normal_points, outlier_points],
    ["Decision Boundary", "Training data", "Subsequent normal points",
        "Subsequent outlier points"],
    bbox_to_anchor = (1.6, 0.6),
    loc="center right",
)
plt.xlabel(
    (f"Errors in training: {n_error_train}/200\n"
     f"Errors novel normal: {n_error_normal}/20\n"
     f"Errors novel outlier: {n_error_outliers}/20"))
plt.show()
