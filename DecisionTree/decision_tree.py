from classifier import classifier
import numpy as np

class decision_tree(classifier):

    def __init__(self, criterion='entropy'):
        if criterion == 'entropy' or criterion == 'gini':
            self.criterion = criterion
            self.tree = dict()
        else:
            raise Exception("criterion has not attribute ", criterion)

    def gini(self, Y):
        size = len(Y)
        counts = dict()
        for y in Y:
            if y not in counts:
                counts[y] = 0.
            counts[y] += 1.
        gini = 0.
        for key in counts:
            prob = counts[key] / size
            gini += prob * (1-prob)
        return gini


    def entropy(self, Y):
        from math import log

        size = len(Y)
        counts = dict()
        for y in Y:
            if y not in counts:
                counts[y] = 0.
            counts[y] += 1.
        entropy = 0.
        for key in counts:
            prob = counts[key] / size
            entropy -= prob * log(prob,2)
        return entropy

    def result(self, Y):
        if self.criterion == 'entropy':
            return self.entropy(Y)
        else:
            return self.gini(Y)

    def split_data(self, X, Y, axis, value):
        return_x = []
        return_y = []

        for x, y in (zip(X.tolist(), Y.tolist())):
            if x[axis] == value:
                reduced_x = x[:axis]
                reduced_x.extend(x[axis+1:])
                return_x.append(reduced_x)
                return_y.append(y)
        return np.array(return_x), np.array(return_y)


    def choose_feature(self, X, Y):
        result = self.result(Y)
        best_information_gain = 0.
        best_feature = -1
        for i in range(len(X[0])):  # For each feature
            feature_list = [x[i] for x in X]
            values = set(feature_list)
            result_i = 0.
            for value in values:
                sub_x, sub_y = self.split_data(X, Y, i, value)
                prob = len(sub_x) / float(len(X))
                result_i += prob * self.result(sub_y)
            info_gain = result - result_i
            if info_gain > best_information_gain:
                best_information_gain = info_gain
                best_feature = i
        return best_feature


    def class_dict(self, Y):
        classes = dict()
        for y in Y:
            if y not in classes:
                classes[y] = 0
            classes[y] += 1
        return classes


    def majority(self, Y):
        from operator import itemgetter
        # Use this function if a leaf cannot be split further and
        # ... the node is not pure
        classcount = self.class_dict(Y)
        sorted_classcount = sorted(classcount.items(), key=itemgetter(1), reverse=True)
        return sorted_classcount[0][0]


    def build_tree(self, X, Y):
        # IF there's only one instance or one class, don't continue to split
        if len(Y) <= 1 or len(self.class_dict(Y)) == 1:
            return Y[0]

        if len(X[0]) == 1:
            return self.majority(Y)   # TODO: Fix this

        best_feature = self.choose_feature(X, Y)
        if best_feature < 0 or best_feature >= len(X[0]):
            return self.majority(Y)

        this_tree = dict()
        feature_values = [example[best_feature] for example in X]
        unique_values = set(feature_values)
        for value in unique_values:
            # Build a node with each unique value:
            subtree_x, subtree_y = self.split_data(X, Y, best_feature, value)
            if best_feature not in this_tree:
                this_tree[best_feature] = dict()
            if value not in this_tree[best_feature]:
                this_tree[best_feature][value] = 0
            this_tree[best_feature][value] = self.build_tree(subtree_x, subtree_y)
        return this_tree


    def fit(self, X, Y):
        self.tree = self.build_tree(X, Y)
        

    def predict(self, X):
        hyps = np.array([])
        for x in X:
            hyps = np.append(hyps, [self.predict_sub(self.tree, x)])
        return hyps
            

    def predict_sub(self, tree, x):
        for k, v in tree.items():
            if x[k] not in v:
                return -1
            if isinstance(v[x[k]], dict):
                return self.predict_sub(v[x[k]], np.append(x[:k], x[k+1:]))
            else:
                return v[x[k]]
