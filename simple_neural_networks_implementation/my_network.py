import math
import random
import numpy as np
import neural_data


# helper functions
def sigma(x):
    return 1 / (1 + np.exp(-x))


# class network

class Network:
    """
    A neural Network.

    Neural Network takes some numbers as data for every traing exmaple and learn the best parametsr for it
    by gradiant decent and back_propagation and eventullay comes up with best weights and biaaess for
    the nurons in every layer and can then predict labl for the future data input.

    Parameters
    ----------
    error_of_calculation : float
       error of diffrence between the desired output and the resultant output each time the learning procedure
       takes place. This indictes when the learning should stop and take the resultant weights and biases
       as the end reasults that nn classifier can use from then on.
    learning_rate : float
       Rate on which the cost function calculates next weights and biases in gradiant decent.
    num_of_layers : integer
       Number of layers that Network should have.
    list_of_nurons_per_layer: List of integers
        List of number of nurons that every layer in Network should have.
    data_file_name : String
        Name of the file containg the data to be used for network learing process.

    first_layer_num_of_nurons: integer
        number of nurons in the first layer.


    Examples
    --------
    To define the a neural network with 24 as initial number of nurons in first layer and other attribuites as
    described above, we can instantiate it as :

    >>> my_network = Network(0.001,0.01, 4, [10, 10, 10, 2], "file.csv",24)
    """

    open_file_data = None

    def __init__(self, error_of_calculation, learning_rate, num_of_layers, list_of_nurons_per_layer, data_file_name,
                 first_layer_num_of_nurons):

        self.alpha = learning_rate
        self.initial_data_dim = first_layer_num_of_nurons
        self.error = error_of_calculation
        self.num_layers = num_of_layers
        self.num_nurons_lst = list_of_nurons_per_layer
        self.list_of_weights = []
        self.list_of_biases = []
        self.list_of_z_i = []
        self.list_of_a_i = []
        self.label = " "
        self.list_of_trained_w_and_b = {"w": [], "b": []}
        self.list_of_trained = {"w": [], "b": []}
        self.data_file = data_file_name
        self.set_up()
        Network.open_file_data = open(self.data_file, "r")

    def set_up(self):

        """
        Sets up the neural network inclusing openning the data files and setting up the random weights and biases
        for the network to start.
        :return: None
        :rtype: None
        """

        pre = self.initial_data_dim
        for i in range(self.num_layers):
            w = np.random.random((self.num_nurons_lst[i], pre))
            b = np.random.random((self.num_nurons_lst[i], 1))
            pre = self.num_nurons_lst[i]
            self.list_of_biases.append(b)
            self.list_of_weights.append(w)

   


    def read_one_set_of_data(self, name_of_file):

        label = " "
        lst = []
        for i in range(24):
            line = name_of_file.readline()
            if line != "":
                s_line = line.split()
                lst.append(float(s_line[0]))

        line = name_of_file.readline()
        if line != "":
            s = line.split()
            label = s[2]
        return label, lst

    def forward(self, X, y, index, layer, num_nurons):

        lst = []
        w = self.list_of_weights[layer]
        b = self.list_of_biases[layer]
        y_inter = w.dot(X) + b
        y = sigma(y_inter)
        lst.append(y)
        self.list_of_a_i.append(y)
        self.list_of_z_i.append(y_inter)
        return y

    def main_trainer(self, X, pre_num_nurons, num_of_layers):

        y = []
        X = np.array(X).reshape((24, 1))
        index = 0
        layer = 0

        for t in range(num_of_layers):
            num_nurons = self.num_nurons_lst[t]
            y = self.forward(X, y, index, layer, num_nurons)
            index = index + 1
            layer = layer + 1
            X = y
        return y

    def derivative(self, x):

        return sigma(x) * (1 - sigma(x))

    def forward_trained(self, X, y, index, layer, num_nurons):

        lst = []
        w = self.list_of_trained["w"][layer]
        b = self.list_of_trained["b"][layer]
        y_inter = w.dot(X) + b
        y = sigma(y_inter)
        lst.append(y)
        self.list_of_a_i.append(y)
        self.list_of_z_i.append(y_inter)
        return y

    def main_trainer_trained(self, X, pre_num_nurons, num_of_layers):

        y = []
        X = np.array(X).reshape((24, 1))
        index = 0
        layer = 0

        for t in range(num_of_layers):
            num_nurons = self.num_nurons_lst[t]
            y = self.forward(X, y, index, layer, num_nurons)
            index = index + 1
            layer = layer + 1
            X = y
        return y

    def cost_of_ai(self, i, y):

        """
        It calculates the cost of every layer i with regards to its prvious layer strting form layer with matrix values
        y.
        :param i: layer number
        :type i: integer
        :param y: matrix conating the first layer's values
        :type y: numpy array
        :return: a numpy array
        :rtype: Returns the cost assocaited with the ith layer.
        """

        Cost = 0

        for l in range(len(y)):
            Cost = Cost + 2 * (y[l] - self.list_of_z_i)

        return Cost

    def modify_w(self, W):
        index = 0
        row = 0
        col = 0
        sum_m = 0
        n = 0
        i = 0
        t = self.num_layers
        d1 = W[0][index].shape[0]
        d2 = W[0][index].shape[1]
        some_w = np.array(W[0][index])
        while i < t:
            while row < d1:
                while col < d2:
                    for lst in W:
                        sum_m = sum_m + lst[index][row, col]
                        n = n + 1
                    mean = sum_m / n
                    some_w[row, col] = mean
                    col = col + 1
                row = row + 1
                col = 0
                index = index + 1
                d1 = W[0][index].shape[0]
                d2 = W[0][index].shape[1]
                self.list_of_trained.get("w").append(some_w)
                some_w = np.array(W[0][index])
            i = i + 1

    def modify_b(self, W):
        index = 0
        row = 0
        col = 0
        sum_m = 0
        n = 0
        i = 0
        t = self.num_layers
        d1 = W[0][index].shape[0]
        d2 = W[0][index].shape[1]

        some_b = np.array(W[0][index])
        while i < t:

            while row < d1:

                while col < d2:

                    for lst in W:
                        sum_m = sum_m + lst[index][row, col]

                        n = n + 1
                    mean = sum_m / n
                    some_b[row, col] = mean
                    col = col + 1
                row = row + 1
                col = 0

                index = index + 1
                d1 = W[0][index].shape[0]
                d2 = W[0][index].shape[1]
                self.list_of_trained.get("b").append(some_b)
                some_b = np.array(W[0][index])

            i = i + 1

    def back(self, y, desired_y):

        """
        It does backpropagation starting from y as the last layer .
        :param y:
        :type y:
        :return:
        :rtype: tuple of updated matrices of weights and biases.
        """

        y = np.array([1, 0])
        if desired_y == 0:
            com_lst = [1, 0]
        else:
            com_lst = [0, 1]
        p = -2
        main_index = 0
        col_w = []
        col_b = []
        col = self.list_of_weights[p].shape[1]
        C = 0
        index = 0
        w_t = []
        b_t = []
        delat_w = self.list_of_weights[p]
        delta_b = self.list_of_biases[p]

        p_index = 0

        layer_i = self.list_of_a_i[p]

        while p_index + self.num_layers <= 0:
            while p + layer_i <= 0:

                for i in range(len(self.list_of_z_i[p].shape[1])):
                    z_i = self.list_of_z_i[p]
                    a_i = self.list_of_a_i[p]
                    a_t = self.list_of_a_i[p - 1]
                    w_t = self.list_of_weights[p]
                    b_t = self.list_of_biases[p]

                    for y_l in com_lst:
                        C = C + 2 * (y_l - a_i[index, 0])
                        index = index + 1
                    dw = C * self.derivative(z_i[main_index, 0]) * a_t[main_index, 0]
                    db = self.derivative(z_i[main_index, 0]) * a_t[main_index, 0]
                    main_index = main_index + 1

                    col_w.append(dw)
                    col_b.append(db)
                col_w_np = np.array(col_w).reshape((w_t.shape[0], 1))
                col_b_np = np.array(col_b).reshape((b_t.shape[0], 1))
                delat_w[:, p - 1] = col_w_np
                delta_b[:, p - 1] = col_b_np
                p = p - 1

            self.list_of_weights[p_index] = self.list_of_weights[p_index] + (delat_w).dot(self.alpha)
            self.list_of_biases[p_index] = self.list_of_biases[p_index] + (delta_b).dot(self.alpha)
            p_index = p_index - 1

    def training_procudure(self):
        # produce and X
        # while not the end of file
        new_data = self.read_one_set_of_data(Network.open_file_data)

        while len(new_data[1]) != 0:

            lst_of_data = new_data[1]
            label = new_data[0]
            if label == "buy":
                desired_y = 0
            else:
                desired_y = 1
            # cost and back
            # go forwards and train
            y = self.main_trainer(lst_of_data, self.initial_data_dim, self.num_layers)

            # cost and back
            if desired_y == 0:
                c1 = y[0] - 1
                c2 = y[1] - 0
            else:
                c1 = y[0] - 0
                c2 = y[1] - 1
            new_y = [c1, c2]

            while desired_y - y[desired_y] >= self.error:

                self.back(np.array(new_y).reshape(2, 1), desired_y)
                y = self.main_trainer(lst_of_data, self.initial_data_dim, self.num_layers)

            L1 = self.list_of_weights
            L2 = self.list_of_biases

            self.list_of_trained_w_and_b.get("w").append(L1)

            self.list_of_trained_w_and_b.get("b").append(L2)
            new_data = self.read_one_set_of_data(Network.open_file_data)

        W = self.list_of_trained_w_and_b.get("w")
        B = self.list_of_trained_w_and_b.get("b")

        self.modify_w(W)
        self.modify_b(B)

    def predict(self, data):

        y = self.main_trainer_trained(data, self.initial_data_dim, self.num_layers)

        # buy 0 and sell 1
        print(y)

        if y[0] > y[1]:
            return "buy"
        else:
            return "sell"






 # test


if __name__ == "__main__":
    num = 3

    list_of_layers = [10, 10, 2]
    input_data_count = 24
    error = 0.01
    rate_of_learning = 0.01

    file_hourly = "/Users/amir/Desktop/fxtime/cur_hours.csv"
    file_daily = "/Users/amir/Desktop/fxtime/cur_daily.csv"
    data_file = "/Users/amir/Desktop/fxtime/currency_new_version.csv"

    lst_of_new_data = [0.78751, 0.78761, 0.78754, 0.78743, 0.7873, 0.78741, 0.7871,
                       0.78773, 0.78774, 0.78776, 0.78809, 0.78805, 0.78777, 0.78823, 0.78844, 0.78807,
                       0.78814, 0.78814, 0.78817, 0.78834, 0.78824, 0.78821, 0.78825, 0.78833]

 

    # provide the data from two files (file1 shows the daily candels info and file2 shows the hourly
    # candels info)
    neural_data.provide_data(file_daily, file_hourly, data_file)

    print("I am here")

    nn = Network(error, rate_of_learning, num, list_of_layers, data_file, input_data_count)
    # it should print buy or sell
    nn.training_procudure()
    print(nn.predict(lst_of_new_data))
