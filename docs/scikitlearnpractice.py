from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
clf = RandomForestClassifier(random_state=0)
X = [[ 1, 2, 3],
     [11,12,13]]
y = [0,1]
clf.fit(X,y)
z = [[15,16,17],[1,2,3]]
print(clf.predict(z))

pre = [[0,15],[1,-10]]
print (StandardScaler().fit(pre).transform(pre))
