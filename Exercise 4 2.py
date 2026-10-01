#Exercise 4.2 - MLP dengan tiga input
#https://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier

# Tiga input per data (sebelumnya dua), target = logical OR dari ketiga input
X = [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 1., 1.],
     [1., 0., 0.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]]
y = [0, 1, 1, 1, 1, 1, 1, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

# Data uji juga harus memiliki tiga nilai
print(clf.predict([[2., 2., 2.], [-1., -2., -1.]]))
print([coef.shape for coef in clf.coefs_])