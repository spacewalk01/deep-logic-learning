import os
import tensorflow as tf
import numpy as np
from sklearn import mixture

DEFAULT_WEIGHT = 4.0

def cluster_weights(links, threshold):
    """Cluster weights (links) using Gaussian mixture model with EM.

    It also determines the number of clusters using Bayesian information
    criterion and sets the weight of all links in each cluster to the
    average of each cluster's weight.

    Args:
        links: weights of links connected to a single unit.
        threshold: bias value.
    Returns:
        Clustered weights
    """

    weights = np.transpose(np.array([links]))

    # Clustering links
    n = len(links)
    MIN_NUM_SAMPLES = 2
    if n > MIN_NUM_SAMPLES:
        # Fit a mixture of Gaussians with EM
        lowest_bic = np.infty
        bic = []
        for n_components in range(2, n):
            gmm = mixture.GMM(n_components=n_components, covariance_type='full')
            gmm.fit(weights)
            # Bayesian information criterion
            bic.append(gmm.bic(weights))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        # Average weights
        ids = best_gmm.predict(weights)
        unique_ids = list(set(ids))

        for i in unique_ids:
            indices = ids == i
            average_weight = np.sum(links[indices]) / len(links[indices])
            links[indices] = average_weight

        return links, ids
    elif n == 2:
        return links, np.array([0, 1])
    else:
        return links, np.zeros(len(links))

def load_data(filename):
    """Read features and training samples from dataset

    Args:
        filename: file name to load data
    Returns:
        y: labels
        X: Training data
        feature_names: a list of feature names
    """
    file = open(filename, 'rt', encoding='UTF8')

    features = []; X = []; y = []
    for line in file:
        line = line.replace('\n', '')
        row = [s.strip() for s in line.split(',')]
        if not features:
            # The first line is a list of feature names
            features = row
        else:
            # The rest of the lines is training data
            X.append([float(s) for s in row[:-1]])
            # The last column stores labels
            y.append(row[-1])
    file.close()

    return np.array(X), np.transpose(np.array([y])), features

class Predicate:
    """Predicate object

    Attributes:
        name: the name of predicate
        negated: indicates whether the predicate is negated.
    """
    def __init__(self, name, negated=False):
        self.name = name
        self.negated = negated

class Rule:
    """First order rule

    Attributes:
        head: the consequent of the rule
        body: the antecedents of the rule
    """
    def __init__(self, head, body):
        self.head = head
        self.body = body

def load_rules(filename):
    """Load rules from a file

    Create a set of rule objects with head and body
    elements from rule file

    Args:
        filename:

    Returns:
        A list of rules
    """
    def cleanse(str):
        """Sanitize a string rule and remove stopwords
        """
        rep = ['\n', '-', ' ', '.']
        for r in rep:
            str = str.replace(r, '')
        return str

    file = open(filename, 'rt', encoding='UTF8')
    ruleset = []
    for line in file:
        tokens = line.split(':')
        head = Predicate(cleanse(tokens[0]))
        body = []
        for obj in tokens[1].split(','):
            obj = cleanse(obj)
            negated = False
            if obj.startswith('not'):
                negated = True
                obj = obj.replace('not', '')
            predicate = Predicate(cleanse(obj), negated=negated)
            body.append(predicate)
        rule = Rule(head, body)
        ruleset.append(rule)
    file.close()
    return ruleset

def rewrite_rules(ruleset):
    """Scan every rule and rewrite the ones with the same consequents

    It implements Towell's rewritting algorithm. If there is more than one
    rule to consequent, then rewrite it as two rules.

    Args:
        ruleset: a set of rules
    Returns:
        A set of rewritten rules. For example:

        A :- B, C.
        A :- D, E.
        are written as
        A :- A'.
        A' :- B, C.
        A :- A''.
        A'' :- D, E.
    """

    # Dict is a dictionary that stores consequences along with their occurrence
    # Keys are to the consequences (head) and values are the occurrence
    dict = {}
    for rule in ruleset:
        if rule.head.name not in dict:
            dict[rule.head.name] = 1
        else:
            dict[rule.head.name] += 1

    # Rewrite rules that conclude the same consequences
    rewritten_rules = []
    i = len(ruleset)
    for rule in ruleset[:]:
        if dict[rule.head.name] > 1:
            # Create a new intermediate consequent
            new_predicate = Predicate(rule.head.name + str(i))
            # Create two new rules for the consequence and antecedents
            rewritten_rules.append(Rule(rule.head, [new_predicate]))
            rewritten_rules.append(Rule(new_predicate, rule.body))
            ruleset.remove(rule)
            i += 1
    del dict

    return ruleset + rewritten_rules

def get_antecedents(rules):
    """Retrieve all the antecedents from a set of rules

    Args:
        rules: a set of rules.

    Returns:
        all_antecedents: Retrieves and flattens rules' antecedents.
        Only, antecedent names are returned.
    """

    all_antecedents = []
    for rule in rules:
        for predicate in rule.body:
            if predicate.name not in all_antecedents:
                all_antecedents.append(predicate.name)
    return all_antecedents

def get_consequents(rules):
    """Retrieve all the consequents from a set of rules

    Args:
        rules: a set of rules.

    Returns:
        all_consequents: Retrieves and flattens rules' consequents.
        Only, consequent names are returned.
    """

    all_consequents = []
    for rule in rules:
        if rule.head.name not in all_consequents:
            all_consequents.append(rule.head.name)
    return all_consequents

def rule_to_network(ruleset):
    """Translating rules to network (Towell's mapping algorithm)

    Establishes a mapping between a set of rules and a neural network.
    This mapping creates layers, weights and biases for the neural network.

    Args:
        ruleset: a set of rewritten rules.

    Returns:
        weights: network weights
        biases: network biases

        Weights and biases are initialized corresponding to disjunctive and
        conjunctive rules
    """

    # Create network layers from rules
    rule_layers = []
    l = 0
    copied_rules = ruleset.copy()
    while len(copied_rules) > 0:
        if l == 0:
            all_antecedents = get_antecedents(copied_rules)
        else:
            all_antecedents = get_antecedents(rule_layers[-1])

        rule_layer = []
        for rule in copied_rules[:]:
            if rule.head.name not in all_antecedents:
                rule_layer.append(rule)
                copied_rules.remove(rule)
        del all_antecedents[:]
        rule_layers.append(rule_layer)

    # Reverse the order of the list
    rule_layers = rule_layers[::-1]

    # Create weights and biases for each layer in the network
    omega = DEFAULT_WEIGHT
    weights = []
    biases = []
    layers = []
    last_layer = []

    for rule_layer in rule_layers:

        current_layer = get_antecedents(rule_layer)
        next_layer = get_consequents(rule_layer)

        for unit in current_layer:
            if unit not in last_layer:
                last_layer.append(unit)
        current_layer = last_layer.copy()

        layers.extend([current_layer, next_layer])
        last_layer = next_layer.copy()

        # Store the occurrence of consequences. For example,
        # if a consequent occurred more than one, then it is a disjunctive rule
        dict = {}
        for rule in rule_layer:
            if rule.head.name not in dict:
                dict[rule.head.name] = 1
            else:
                dict[rule.head.name] += 1

        weight = np.zeros([len(current_layer), len(next_layer)])
        bias = np.zeros(len(next_layer))

        for rule in rule_layer:

            j = next_layer.index(rule.head.name)
            for predicate in rule.body:
                i = current_layer.index(predicate.name)
                if predicate.negated:
                    weight[i][j] = -omega
                else:
                    weight[i][j] = omega

            if dict[rule.head.name] > 1:
                bias[j] = 0.5 * omega
            else:
                p = len(rule.body)
                bias[j] = (p - 0.5) * omega

        weights.append(np.array(weight))
        biases.append(np.array([bias]))

    return weights, biases, layers

def preprocess_data(dataset, feature_names, layers):
    """Preprocessing input data"""

    last_layer = []; X = []; i = 1
    for layer in layers:
        indices = []
        if i == 1:
            # input layer
            indices = [feature_names.index(unit) for unit in layer]
            X.append(dataset[:, indices])

        elif i % 2 != 0 and len(last_layer) > 0:
            # hidden and output layer
            hidden_input = [unit for unit in layer if unit not in last_layer]
            indices = [feature_names.index(unit) for unit in hidden_input]
            x = dataset[:, indices]
            n = len(x)
            m = len(x[0])
            X.append(x + 0.00001 * np.random.rand(n, m))
        else:
            last_layer = layer
        i += 1

    return X

def eliminate_weights(weights, biases):
    """Eliminate weights that are not contributing to the output"""
    cluster_ids = []
    for i in range(len(weights)):
        cluster = []
        for j in range(weights[i].shape[1]):
            b = biases[i][0, j]
            (_w, ids) = cluster_weights(weights[i][:, j], b)
            weights[i][:, j] = list(_w)
            cluster.append(ids)
        cluster_ids.append(cluster)

    return weights, biases, cluster_ids

def network_to_rule(weights, biases, cluster_indices, layers):
    """Translate network to rule.

    Extract rules from neural network, specifically weights.

    Args:
        weights: a set of weights
        biases: a set of biases
        cluster_indices: a set of indices that clusters weights
        layers: layers of neural network units

    Returns:
        a set of rules extracted from the neural network
    """

    rules = []
    layers = np.array(layers)
    weight_range = range(0, len(weights))
    layer_range = range(0, len(layers), 2)
    for i, l in zip(weight_range, layer_range):
        current_layer = np.array(layers[l])
        next_layer = layers[l + 1]
        for j in range(weights[i].shape[1]):
            b = biases[i][0, j]
            w = weights[i][:, j]
            head = next_layer[j]
            indices = cluster_indices[i][j]
            unique_ids = list(set(indices))
            body = ''
            for id in unique_ids:
                if body != '':
                    body += ' + '
                matched_indices = indices == id
                antecedents = current_layer[matched_indices]
                threshold = w[matched_indices]
                body += str(threshold[0]) + ' * nt(' + ','.join(antecedents) + ')'
            new_rule = head + ' :- ' + str(b) + ' < ' + body
            rules.append(new_rule)

            print(head + ' = 0')
            print('if ' + str(b) + ' < ' + body + ':')
            print('\t' + head + ' = 1')
    return rules

def add_input_units(weights, layers, feature_names):
    """Add input features not referred by the rule set

    This addition is necessary because a set of rules that
    is only approximately correct may not identify every input
    that is required for correctly learning a concept.
    """

    additional_units = feature_names.copy()

    for layer in layers:
        for unit in layer:
            if unit in feature_names:
                additional_units.remove(unit)

    w = weights[0]
    zeros = np.zeros((len(additional_units), w.shape[1]))
    weights[0] = np.row_stack([w, zeros])
    layers[0] += additional_units




    return weights, layers

def add_hidden_units(weights, biases, layers):
    """Add units to hidden layers
    """
    w1 = weights[0]
    w2 = weights[1]
    zeros1 = np.zeros((w1.shape[0], 3))
    weights[0] = np.column_stack([w1, zeros1])
    zeros2 = np.zeros((3, 1))

    weights[1] = np.row_stack([w2, zeros2])
    b = biases[0]
    biases[0] = np.column_stack([b, np.zeros((1, 3))])

    layers[1].insert(len(layers[1]), 'head1')
    layers[1].insert(len(layers[1]), 'head2')
    layers[1].insert(len(layers[1]), 'head3')

    layers[2].insert(len(layers[2]), 'head1')
    layers[2].insert(len(layers[2]), 'head2')
    layers[2].insert(len(layers[2]), 'head3')

    return weights, biases, layers

def simplify_rules(rules):

    return rules

def save(rules, filepath):
    with open(filepath, 'wb') as f:
        for row in rules:
            f.write(repr(str(row)) + '\n')

class KBANN:
    """Knowledge base artificial neural network

    Create KBANN network on the tensorflow framework.

    Attributes:
        weights: a set of tensors containing all weights in the network
        biases: a set of tensors containing all biases
        num_layers: the number of layers in the network
        input_data: a set of tensors containing input data in each layer.
            If there is no input for the layer, it stores an empty list
        input_mask: a list of booleans that indicates in which layer it feeds the network
        learning_rate: learning rate for optimization
    """

    def set_weights(self, input_weights, fixed=False):
        if fixed:
            fixed_tensors = []
            for w in input_weights:
                tensor = tf.constant(w, dtype=tf.float32)
                fixed_tensors.append(tensor)
            return fixed_tensors
        else:
            tensors = []
            for w in input_weights:
                #tensor = tf.Variable(tf.random_uniform([len(w), len(w[0])], -1, 1), dtype=tf.float32)
                tensor = tf.Variable(w + 0.1 * np.random.rand(len(w), len(w[0])), dtype=tf.float32)
                tensors.append(tensor)
            return tensors

    def set_biases(self, input_biases):
        tensors = []
        for b in input_biases:
            #tensor = tf.Variable(tf.random_uniform([len(b), len(b[0])], -1, 1), dtype=tf.float32)
            tensor = tf.Variable(b + 0.1 * np.random.rand(1, len(b)), dtype=tf.float32)
            tensors.append(tensor)
        return tensors

    def __init__(self, weights, biases, X, y, learning_rate):
        """Set network parameters"""

        self.weights = self.set_weights(weights)
        self.biases = self.set_biases(biases)
        self.num_layers = len(weights)
        self.input_data = []
        self.input_mask = []
        self.learning_rate = learning_rate
        for x in X:
            self.input_data.append(tf.constant(x, dtype=tf.float32))
            if x.shape[1] > 0:
                self.input_mask.append(True)
                print('masked')
            else:
                self.input_mask.append(False)

        self.targets = tf.constant(y, dtype=tf.float32)

    def propagate_forward(self):
        """Implements the forward propagation"""

        activations = [tf.sigmoid(tf.matmul(self.input_data[0], self.weights[0]) - self.biases[0])]
        for i in range(1, self.num_layers):
            if self.input_mask[i]:
                input_tensor = tf.concat(1, [activations[-1], self.input_data[i]])
            else:
                input_tensor = activations[-1]
            activation = tf.sigmoid(tf.matmul(input_tensor, self.weights[i]) - self.biases[i])
            activations.append(activation)
        return activations[-1]

    def propagate_forward_with_dropout(self):
        """Implements the forward propagation"""

        activations = [tf.sigmoid(tf.matmul(self.input_data[0], self.weights[0]) - self.biases[0])]

        for i in range(1, self.num_layers):
            if self.input_mask[i]:
                input_tensor = tf.concat(1, [activations[-1], self.input_data[i]])
            else:
                input_tensor = activations[-1]
            dropout_layer = tf.nn.dropout(input_tensor, 0.8)
            activation = tf.sigmoid(tf.matmul(dropout_layer, self.weights[i]) - self.biases[i])
            activations.append(activation)
        return activations[-1]

    def compute_loss(self, logits):
        return tf.reduce_mean(-self.targets * tf.log(logits) - (1.0 - self.targets) * tf.log(1.0 - logits))

    def train(self, cost):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        return optimizer.minimize(cost)

def display(arrays):
    for array in arrays:
        print(array)

if __name__ == "__main__":

    CURRENT_DIRECTOR = os.getcwd()

    # Initial parameters
    training_epochs = 2000000
    display_step = 10000
    learning_rate = 0.1

    # Load training data
    data_file_path = CURRENT_DIRECTOR + '/Datasets/student.txt'
    X, y, feature_names = load_data(data_file_path)

    # Translate rules to a network
    rule_file_path = CURRENT_DIRECTOR + '/Datasets/student_rules.txt'
    ruleset = load_rules(rule_file_path)
    ruleset = rewrite_rules(ruleset)
    weights, biases, layers = rule_to_network(ruleset)
    display(layers)
    print('---------------------')
    # Add input features not referred by the rule set
    #weights, layers = add_input_units(weights, layers, ['complete_course'])
    weights, layers = add_input_units(weights, layers, ['complete_course', 'freshman', 'sent_application', 'high_gpa'])

    # Add hidden units not specified by the initial rule set
    weights, biases, layers = add_hidden_units(weights, biases, layers)
    display(layers)

    # Pre-process input data
    X = preprocess_data(X, feature_names, layers)

    print('Parameters 0:')
    display(weights)
    display(biases)

    # Construct a training model
    model = KBANN(weights, biases, X, y, learning_rate) # tensorflow model

    # Launch the graph
    with tf.Session() as session:

        # Initialize all variables
        session.run(tf.initialize_all_variables())

        logits = model.propagate_forward_with_dropout()
        loss = model.compute_loss(logits)
        train_op = model.train(loss)
        weights = session.run(model.weights)
        biases = session.run(model.biases)

        # Refine rules
        for epoch in range(training_epochs):
            session.run(train_op)
            if epoch % display_step == 0:
                loss_value = session.run(loss)
                print('Epoch %d: Loss = %.9f' % (epoch / display_step, loss_value))
        print('Rule Refinement Finished!')

        weights = session.run(model.weights)
        biases = session.run(model.biases)

        print('Parameters 2:')
        display(weights)
        display(biases)

        weights, biases, cluster_indices = eliminate_weights(weights, biases)
        print('Parameters 3:')
        display(weights)
        display(biases)

        model.set_weights(weights, fixed=True)
        model.set_biases(biases)
        logits = model.propagate_forward()
        loss = model.compute_loss(logits)
        train_op = model.train(loss)

        # Optimize bias
        for epoch in range(1000000):
            _, loss_value = session.run([train_op, loss])
            if epoch % display_step == 0:
                print('Epoch %d: Loss = %.9f' % (epoch / display_step, loss_value))
        biases = session.run(model.biases)

        # Translate network to rules
        ruleset = network_to_rule(weights, biases, cluster_indices, layers)
        print('Parameters 4:')
        display(weights)
        display(biases)

        # save(ruleset, CURRENT_DIRECTOR + '/Datasets/extracted_rules.txt')
        print('Rule Extraction Finished!')
