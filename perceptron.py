import random
import inspect
import uuid

# @Author: rchromik

# @Title: Simple example of Machine learning - Perceptron (https://en.wikipedia.org/wiki/Perceptron)
# simplest model of neural network - that contains just one neuron.

# @Description: Problem is simple - classify one point if it's about the y = x line or below,
#   if it's above than points is labeled as True, otherwise (below) is labeled as False;
#
#   Steps:
#   1) Take input and "weight it"
#   2) Take the sum in the activation function -> defines the label(in this case True or False)
#   3) Calculate the error (desired output - obtained)
#   4) Adjust the weights (using error*input*learning_rate formula)
#   5) Repeat a vary lot of times (the more, the better)


class Neuron:
    def __init__(self, synapses_count):
        # store information about input synapses count
        self.synapses_count = synapses_count
        # default initialization of weights
        self.weights = []
        for set_feature_index in range(self.synapses_count):
            self.weights.extend([0])

        # learning ratio (how
        self.learning_rate = 0.01

    def get_output_potential(self, p_input_synapses_values):
        output_potential = 0
        for synapse_index in range(self.synapses_count):
            output_potential += p_input_synapses_values[synapse_index] * self.weights[synapse_index]
        return output_potential

    def get_output_function_result(self, p_input_synapses_values, to_call_function):

        if len(inspect.getargspec(to_call_function).args) != 1:
            raise Exception("Output function must have one parameter!")

        output_potential = self.get_output_potential(p_input_synapses_values)
        return to_call_function(output_potential)

    def reset_weights(self):
        self.weights = [0] * len(self.weights)


class Perceptron:
    def __init__(self):
        self.name = str(uuid.uuid4())
        self.neuron = Neuron(2)
        print('[Perceptron ' + self.name + '] was borned.')

    def __del__(self):
        print('[Perceptron ' + self.name + '] was killed.\n')

    @staticmethod
    def is_axon_activated(p_output_potential):
        if p_output_potential > 0:
            return 1
        else:
            return 0

    def forget_everything(self):
        self.neuron.reset_weights()
        print('[Perceptron ' + self.name + '] forgot everything he learned...')

    def train_random_data(self, p_length):
        random_data = generate_random_training_data(p_length)
        self.train(random_data)

    def train(self, p_training_set):
        neuron = self.neuron

        for example in p_training_set:
            expected_result_index = len(example) - 1

            if len(example) != neuron.synapses_count + 1:
                raise Exception('Invalid input data!')
            # predict result
            data_for_synapses = []
            for set_feature_index in range(neuron.synapses_count):
                data_for_synapses.extend([example[set_feature_index]])

            # compare predicted result with correct data and count error,
            # last index of training set record is correct result
            # output_potential = neuron.get_output_potential(data_for_synapses)
            is_output_activated = neuron.get_output_function_result(data_for_synapses, self.is_axon_activated)
            is_example_activated = example[expected_result_index]
            error = is_example_activated - is_output_activated

            # learning process (adjustment of weights)
            for set_feature_index in range(neuron.synapses_count):
                neuron.weights[set_feature_index] += error * example[set_feature_index] * neuron.learning_rate

        print('[Perceptron ' + self.name + '] was trained on data with ' + str(len(p_training_set)) + ' examples.')

    def predict(self, p_inputs):
        return self.neuron.get_output_function_result(p_inputs, self.is_axon_activated)

    # Testing process - show what Perceptron learned -> generate testing data
    # and compare them with Perceptron's predictions
    def solve_random_test(self, test_length):
        correct_answers_count = 0
        for i in range(test_length):
            x = random.random() * 1000
            y = random.random() * 1000
            correct_answer = get_correct_answer(x, y)
            predicted_result = self.predict([x, y, correct_answer])
            if predicted_result == correct_answer:
                correct_answers_count += 1

        print('[Perceptron ' + self.name + '] solved randomly generated data set with ' + str(test_length) + ' rows with ' + str(
            ((correct_answers_count * 100) / test_length)) + ' % success.')


# Returns correct answer, used for training data generating
# and for comparision of predicted result with correct one
def get_correct_answer(p_x, p_y):
    if p_x > p_y:
        return 1
    else:
        return 0


# Create new Perceptron
perceptron = Perceptron()


# Preparing training data for Perceptron
def generate_random_training_data(training_data_lenght):
    training_set = []
    for i in range(training_data_lenght):
        x = random.random() * 1000
        y = random.random() * 1000
        training_set.append([x, y, get_correct_answer(x, y)])
    return training_set


# Learning process - Perceptron is learning from learning set
perceptron.train_random_data(15)
perceptron.solve_random_test(100)
perceptron = {}

perceptron = Perceptron()
perceptron.solve_random_test(100)
perceptron.train_random_data(3000)
perceptron.solve_random_test(100)


perceptron = Perceptron()
for i in range(5):
    perceptron.train_random_data(5)
    perceptron.solve_random_test(100)
perceptron.forget_everything()
perceptron.solve_random_test(100)
for i in range(5):
    perceptron.train_random_data(5)
    perceptron.solve_random_test(100)
