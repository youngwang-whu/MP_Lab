# -*- coding=utf-8 -*-
# Author: Y'A'Wang
# Date: 07.29.2018
# Aim：Use scikit to class plot of different energy
# Highlight: Pycharm查看源码的方法：
# 将光标移动至要查看的方法处，按住ctrl  点击鼠标左键，即可查看该方法的源码。


from __future__ import absolute_import, division, print_function

import numpy as np
import xlrd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import itertools



# Read xls files
ExcelFile1 = xlrd.open_workbook('140.xlsx')
ExcelFile2 = xlrd.open_workbook('142.xlsx')
ExcelFile3 = xlrd.open_workbook('144.xlsx')
ExcelFile4 = xlrd.open_workbook('146.xlsx')
ExcelFile5 = xlrd.open_workbook('148.xlsx')

# load data
def load_data(ExcelFile):
    # Get the contents of sheet
    sheet = ExcelFile.sheet_by_name('Sheet1')
    # Get the whole value of sheet
    # Highlight:通过列表索引的方法，简单解决了读取excel列表时产生空格的问题
    cols = []
    for i in np.arange(sheet.ncols):
        cols.append(sheet.col_values(i)[1:])
    return np.array(cols[0:20]), np.array(cols[20:])
activity_140, dose_140 = load_data(ExcelFile1)
activity_142, dose_142 = load_data(ExcelFile2)
activity_144, dose_144 = load_data(ExcelFile3)
activity_146, dose_146 = load_data(ExcelFile4)
activity_148, dose_148 = load_data(ExcelFile5)

X = np.concatenate((activity_140, activity_142, activity_144, activity_146, activity_148), axis=0)
Y = np.concatenate((np.ones([20, 1])*140, np.ones([20, 1])*142, np.ones([20, 1])*144,
                    np.ones([20, 1])*146, np.ones([20, 1])*148), axis=0)


x1 = X*(np.ones((100, 302)) + np.random.randn(100, 302)*0.05)
x2 = X*(np.ones((100, 302)) + np.random.randn(100, 302)*0.08)
x3 = X*(np.ones((100, 302)) + np.random.randn(100, 302)*0.1)
x4 = X*(np.ones((100, 302)) + np.random.randn(100, 302)*0.2)
x5 = X*(np.ones((100, 302)) + np.random.randn(100, 302)*0.3)
x6 = X*(np.ones((100, 302)) + np.random.randn(100, 302)*0.4)


x = np.concatenate((X, x1, x2, x3, x4, x5, x6), axis=0)
y = np.concatenate((Y, Y, Y, Y, Y, Y, Y), axis=0)
n_samples, n_features = x.shape

# 标签二值化, 转换成0，1矩阵
y = label_binarize(y, classes=[140, 142, 144, 146, 148])
n_classes = y.shape[1]

# shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.5,random_state=0)
X_train = x
y_train = y
X_test = X*(np.ones((100, 302)) + np.random.randn(100, 302)*0.35)
y_test = label_binarize(Y, classes=[140, 142, 144, 146, 148])

# ‘hidden_layer_sizes=(h1, h2, …… , hn)'中h1~hn分别代表hidden layers 每层上的神经元个数(不算input和output两层)，
#  length=n=网络总层数-2
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, verbose=True, max_iter=1000,
                    hidden_layer_sizes=(200, 20), random_state=1)

probas = clf.fit(x, y).predict_proba(X_test)


# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], probas[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), probas.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)

# Predicting & Confusion matrix
pred_y = clf.predict(X_test)
pred_label = np.argmax(pred_y, axis=1)
true_label = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(true_label, pred_label)
print(conf_matrix)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix, classes=np.array(['E=140MeV', 'E=142MeV', 'E=144MeV', 'E=146MeV','E=148MeV']),
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix, classes=np.array(['140MeV', '142MeV', '144MeV', '146MeV','148MeV']),
                      normalize=True, title='Normalized confusion matrix')

plt.show()





# 加载保存在磁盘中的分类器
plt.savefig（'confusion_matrix.jpg')
# clf = joblib.load('class_energy.pkl')
# joblib.dump(clf, 'class_energy.pkl')




def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


