import pandas as pd
import math
import operator

def calc_entropy(data_set):
    #the data_set is a list
    num_entries = len(data_set)

    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    entropy = 0.0

    for key in label_counts.keys():
        prob = float(label_counts[key]) / num_entries
        entropy -= prob * math.log(prob, 2)

    return entropy    


def split_data_set(data_set, col, value):
    #split the data with specific index equaling value and then remove this column
    re_data_set = []
    for feat_vec in data_set:
        if feat_vec[col] == value:
            reduced_feat_vec = feat_vec[:col]
            reduced_feat_vec.extend(feat_vec[col+1:])
            re_data_set.append(reduced_feat_vec)
    return re_data_set

def choose_best_feat_to_split(data_set):
    num_feat = len(data_set[0]) - 1
    base_entropy = calc_entropy(data_set)
    best_info_gain = 0.0
    best_feat = -1
    for i in range(num_feat):
        feat_list = [example[i] for example in data_set]
        unique_values = set(feat_list)
        new_entropy = 0.0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat = i
    return best_feat

def majority(class_list):
    #Count the majorty of this feature
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

def create_tree(data_set, labels):
    class_list = [example[-1] for example in data_set]

    if class_list.count(class_list[0]) == len(class_list):
        #stop splitting if all the classes are equal
        return class_list[0]
    if len(data_set[0]) == 1:
        #stop splitting if there is no features in data_set
        return majority(class_list)

    best_feat = choose_best_feat_to_split(data_set)
    best_feat_label = labels[best_feat]
    tree = {best_feat_label:{}}
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_labels = labels[:]
        tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return tree

def classify(tree, labels, data):
    root = tree.keys()[0]
    left_tree = tree[root]
    feat_index = labels.index(root)
    key = data[feat_index]
    value_of_feat = left_tree[key]
    if isinstance(value_of_feat, dict):
        class_label = classify(value_of_feat, labels, data)
    else:
        class_label = value_of_feat
    return class_label
    
def classify_all(tree, labels, data_set):
    class_labels = []
    i = 0
    for data in data_set:
        class_label = classify(tree, labels, data)
        class_labels.append(class_label)
        #print i
        i += 1
    return class_labels

def get_dataset(file_path, is_train):
    df = pd.read_csv(file_path)
    if is_train:
        df = df[['Sex', 'Pclass', 'Survived']]
        labels = list(df.columns)
        labels.pop(-1)
        data_set = df.values.tolist()
    else:
        df = df[['Sex', 'Pclass']]
        labels = list(df.columns)
        data_set = df.values.tolist()
    return data_set, labels

def getResult(train_file_path, test_file_path):
    train_data, train_labels = get_dataset(train_file_path, True)
    test_data, test_labels = get_dataset(test_file_path, False)
    
    
    tree = create_tree(train_data, train_labels)
    print tree

    print classify_all(tree, test_labels, test_data)

if __name__=='__main__':
    #It is from kaggle Titanic project
    getResult('./train.csv', './test.csv')
    
