import csv
import math
import random


class Node:
    def __init__(self,
                 condition,  # used for print condition
                 feature_index,  # used in detect_label to find corresponding value from test data
                 value,  # used in detect_label to compare with test data value
                 true_child_data,  # holds the records in true child used for print
                 false_child_data,  # holds the records in false child used for print
                 true_child_node,  # holds true node
                 false_child_node,  # holds false node
                 is_leaf,
                 predict):
        self.condition = condition
        self.feature_index = feature_index
        self.value = value
        self.true_child_data = true_child_data
        self.false_child_data = false_child_data
        self.true_child_node = true_child_node
        self.false_child_node = false_child_node
        self.is_leaf = is_leaf
        self.predict = predict


def read_data(file_address):
    data = []
    with open(file_address) as csvfile:
        input = csv.reader(csvfile)
        for row in input:
            data.append(row)
    return data


def get_train_test_data(data):
    train_set = []  # 2/3 of dataset as train set
    test_set = []  # 1/3 of dataset as test set
    train_counter = 0

    # setting train data
    for train_counter in range(math.ceil(len(data) * 0.66)):
        train_set.append(data[train_counter])

    # setting test data
    for j in range(train_counter+1, len(data)):
        test_set.append(data[j])

    return train_set, test_set


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def find_average(first_num, second_num):
    return (float(first_num) + float(second_num)) / 2.0


def find_unique_values(data, feature):
    values = []
    for row in data:
        value = row[feature]
        if value not in values:
            values.append(value)
    return values


def find_child(data, feature, value, is_digit):
    true_child = []
    false_child = []
    for row in data:
        if is_digit:
            if float(row[feature]) >= value:
                true_child.append(row)
            else:
                false_child.append(row)
        else:
            if value == row[feature]:
                true_child.append(row)
            else:
                false_child.append(row)
    return true_child, false_child


def label_count(data):
    count = {}
    for row in data:
        label = row[-1]
        if label not in count:
            count[label] = 1
        else:
            count[label] += 1
    return count


def entropy(data):
    entropy = 0
    count = label_count(data)
    total = len(data)
    for countValue in count.values():
        entropy -= (countValue / total) * math.log2(countValue / total)
    return entropy


def entropy_split(true_child, false_child):
    percent_true = float(len(true_child) / (len(true_child) + len(false_child)))
    percent_false = float(len(false_child) / (len(true_child) + len(false_child)))
    e_split = percent_true * entropy(true_child) + percent_false * entropy(false_child)
    print('entropy_true:', entropy(true_child), '\nentropy_false:', entropy(false_child), '\nentropy_split:', e_split)
    return e_split


def gain(parent_entropy, true_child, false_child):
    information_gain = parent_entropy - entropy_split(true_child, false_child)
    print('entropy_parent:', parent_entropy, '\ninformation_gain:', information_gain)
    return information_gain


def gain_ratio(information_gain, n_parent, n_true_child, n_false_child):
    ratio = 0.0
    if information_gain != 0:
        split_info = (-1 * ((n_true_child / n_parent) * math.log2(n_true_child / n_parent))) + (-1 * ((n_false_child / n_parent) * math.log2(n_false_child / n_parent)))
        ratio = information_gain / split_info
    print('gain ratio:', ratio, '\n\n')
    return ratio


def split(data):
    parent_entropy = entropy(data)
    best_gain_ratio = 0
    best_node = None

    for feature in range(0, len(headers) - 1):

        values = find_unique_values(data, feature)
        print('------------------------------')
        print('\tfeature:', headers[feature])
        print('\tvalues:', values)
        print('------------------------------')

        values_length = len(values)
        is_digit = is_number(values[0])

        if is_digit:
            values.sort()
            values_length -= 1  # because we are using average of two records

        if values_length == 0:
            print('all the records have similar attribute values.\n\n')

        for i in range(0, values_length):
            if is_digit:
                value = find_average(values[i], values[i + 1])
                node = Node(headers[feature] + ' >= ' + str(value), feature, value, [], [], None, None, False, None)
            else:
                value = values[i]
                node = Node(headers[feature] + ' == ' + str(value), feature, value, [], [], None, None, False, None)

            node.true_child_data, node.false_child_data = find_child(data, feature, value, is_digit)
            print('condition:', node.condition, '\nn_true_child:', len(node.true_child_data), '\nn_false_child:', len(node.false_child_data))
            information_gain = gain(parent_entropy, node.true_child_data, node.false_child_data)
            current_gain_ratio = gain_ratio(information_gain, len(data), len(node.true_child_data), len(node.false_child_data))

            if current_gain_ratio >= best_gain_ratio:
                best_gain_ratio = current_gain_ratio
                best_node = node

    return best_node, best_gain_ratio


def make_tree(data):
    node, gain_ratio_value = split(data)

    # checks leaf node with itself so gain ration becomes zero
    if gain_ratio_value == 0:
        print('-> ', node.condition, ' selected as a leaf.\n')
        return Node(None, [], [], None, None, None, None, True, label_count(data))
    else:
        print('-> ', node.condition, ' with gain ratio:', gain_ratio_value, ' selected.\n')
        true_child_node = make_tree(node.true_child_data)
        false_child_node = make_tree(node.false_child_data)

    return Node(node.condition, node.feature_index, node.value, node.true_child_data, node.false_child_data, true_child_node, false_child_node, False, None)


def print_tree(decision_tree, space=''):
    if decision_tree.is_leaf:
        print(space, decision_tree.predict)
    else:
        print(space, decision_tree.condition)
        print(space, 'true: ')
        print_tree(decision_tree.true_child_node, space + '\t')
        print(space, 'false: ')
        print_tree(decision_tree.false_child_node, space + '\t')


def get_probability(predict):
    probablity = {}
    total = 0

    for value in predict.values():
        total += value

    for key, value in predict.items():
        probablity[key] = int((value / total) * 100)

    return probablity


def select_predict(p, actual):  # selects maximum probability
    maximum_p = 0
    maximum_p_label = ''

    print('actual:', actual, '| predictions(%):', p)

    for label, p in p.items():
        if p > maximum_p:
            maximum_p = p
            maximum_p_label = label
        elif p == maximum_p:
            if random.choice([0, 1]) == 1:
                maximum_p = p
                maximum_p_label = label

    print('-> so it is <<', maximum_p_label, '>> with probability of <<', maximum_p, '% >> ')


def detect_label(testset_row, node):
    test_label = testset_row[-1]

    if node.is_leaf:
        probablity = get_probability(node.predict)
        # select_predict(probablity, test_label) # this method chooses the most probability
        for label, p in probablity.items():
            print('-> it is <<', label, '>> with probability of <<', p, '% >> | actual is:', test_label)
        return

    print('is', node.condition, '?')
    test_value = testset_row[node.feature_index]
    if is_number(test_value):
        if float(test_value) >= node.value:
            print('yes')
            detect_label(testset_row, node.true_child_node)
        else:
            print('no')
            detect_label(testset_row, node.false_child_node)
    else:
        if test_value == node.value:
            print('yes')
            detect_label(testset_row, node.true_child_node)
        else:
            print('no')
            detect_label(testset_row, node.false_child_node)


if __name__ == '__main__':
    dataset = read_data('tennisDataset.csv')
    headers = dataset[0]
    del dataset[0]  # remove headers from data

    train_data, test_data = get_train_test_data(dataset)

    print('\ntrain data (', len(train_data), ') :', train_data)
    print('test data (', len(test_data), ') :', test_data, '\n')

    tree = make_tree(train_data)
    print('********************\n')

    print_tree(tree)

    print('\n********************\n')

    print('predicting started:\n')
    for row in test_data:
        print('------------------------------')
        print(row, 'selected.')
        print('------------------------------')
        detect_label(row, tree)
        print('\n\n')