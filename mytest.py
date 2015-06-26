#coding:utf-8
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./dataset/train.csv')
print data.head(5)
#labels= data.head(5).feature_557.values
labels= data.activity.values
#data.drop([1,2], axis=0)
data=data.drop(['ID','activity'], axis=1)
print data.ix[0:10]
plt.figure(figsize = (18,9))
plt.hist(labels, bins=6)
plt.show()

X=np.array(data)
Y=np.array(labels)

#clf = LogisticRegression()
#clf = tree.DecisionTreeClassifier()
#clf=Lasso(alpha=0.1)
#Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
 #  normalize=False, positive=False, precompute=False, random_state=None,
#   selection='cyclic', tol=0.0001, warm_start=False)
#clf = svm.SVC()
#clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
#gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
#shrinking=True, tol=0.001, verbose=False)





X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,Y,test_size=0.1,random_state=0)
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
#clf = RandomForestClassifier(n_estimators=20)
clf = svm.LinearSVC()


#a=sum(clf.predict(X_test)!=y_test)/float(len(y_test))
#b=np.linalg.norm(clf.predict(X_test)-y_test,1)/float(len(y_test))
#print clf.coef_,sum(clf.predict(X_test)!=y_test),a,b,len(y_test)


'''
pca
'''
n=25
pca=PCA(n_components=n)
pca.fit(X_train)
#for n in range(len(pca.explained_variance_ratio_)):
 #   if 1:
  #      print n,sum(pca.explained_variance_ratio_[:n])/sum(pca.explained_variance_ratio_)
X_train=pca.transform(X_train)



clf.fit(X_train,y_train)
#print clf.predict(pca.transform(X_test))


X_test=pd.read_csv('./dataset/test.csv')
X_test=X_test.drop(['ID'],axis=1)

#-----pca
X_test=pca.transform(X_test)
print X_test.size
#-----
#predictions=clf.predict(X_test)
predictions=clf.predict(X_test)


plt.figure(figsize = (18,9))
plt.hist(predictions, bins=6)
plt.show()
result = pd.read_csv('./dataset/sampleSubmission.csv')
result['activity']=predictions
result.to_csv('./dataset/pca1_svmLearnSVC.csv',index=False)

