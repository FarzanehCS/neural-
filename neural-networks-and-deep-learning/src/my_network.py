import math
import random
import numpy as np


def sigma(x):
    return 1 / (1 + np.exp(-x))


def reader_hourly(line, file_write):
    splitted = line.split()
    open_price = splitted[2]
    high_price = splitted[3]
    low_price = splitted[4]
    close_price = splitted[5]

    if open_price <= close_price:
        result = (round(sigma(float(open_price)), 5), "buy")
    else:
        result = (round(sigma(float(open_price)), 5), "sell")

    file_write.write(str(result[0]) + " " + str(result[1]) + "\n")
    # file_write.write(str(result[1]) + "\n")

    return result


def reader_daily(line, write_to_file):
    # print("line is" + line + "\n")
    splitted = line.split()
    # print(splitted)
    open_price = splitted[1]
    high_price = splitted[2]
    low_price = splitted[3]
    close_price = splitted[4]
    # print(open_price)
    # print(close_price)

    if open_price <= close_price:
        result = (round(sigma(float(open_price)), 5), "buy")
    else:
        result = (round(sigma(float(open_price)), 5), "sell")

        write_to_file.write("OpenDaily" + " " + str(result[0]) + " " + str(result[1]) + "\n")
        # write_to_file.write(str(result[1]) + "\n")
    return result


# Class network

class network:
    file_read_daily = None
    file_read_hourly = None
    open_file_write = None
    open_file_data_all = None

    def __init__(self, error_of_calculation, num_of_layers, list_of_nurons_per_layer, data_file_list, x):

        self.alpha = 0.01
        # self.set_up(data_file_list)
        # self.provide_data(network.file_read_daily, network.file_read_hourly, network.open_file_write)
        self.initial_data_dim = x
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
        self.data_file = data_file_list
        self.set_up(data_file_list)
        # self.provide_data(network.file_read_daily, network.file_read_hourly, network.open_file_write)
        network.open_file_data_all = open(self.data_file[2], "r")

    def set_up(self, data_file_list):

        network.file_read_daily = open(data_file_list[0])
        network.file_read_hourly = open(data_file_list[1])
        # network.open_file_write = open(data_file_list[2], "w")

        pre = self.initial_data_dim
        for i in range(self.num_layers):
            # print(self.num_layers)
            w = np.random.random((self.num_nurons_lst[i], pre))
            b = np.random.random((self.num_nurons_lst[i], 1))
            pre = self.num_nurons_lst[i]
            self.list_of_biases.append(b)
            self.list_of_weights.append(w)

    def refresh(self):

        network.file_read_daily.close()
        network.file_read_hourly.close()
        network.open_file_write.close()
        network.file_read_daily = open(self.data_file_list[0])
        network.file_read_hourly = open(self.data_file_list[1])
        network.open_file_write = open(self.data_file_list[2], "w")

    def provide_data(self, daily, hourly, open_file_write):

        line = daily.readline()
        line = hourly.readline()
        # line = file_read_hourly.readline()
        lst = []
        # label = ""
        while line != "":

            for i in range(24):
                line = hourly.readline()
                if line != "":
                    # print(line + "this is form hours")
                    res = reader_hourly(line, open_file_write)
                    lst.append(float(res[0]))
            line = daily.readline()
            # print("print of line daily is .........." + line + "\n")
            if len(line) != 0:
                res2 = reader_daily(line, open_file_write)
        open_file_write.close()

        # line = file_read_hourly.readline()
        network.open_file_data_all = open(self.data_file[2], "r")

    def read_one_set_of_data(self, name_of_file):

        label = " "
        lst = []
        for i in range(24):
            line = name_of_file.readline()
            if line != "":
                # print("line is ...." + line)
                s_line = line.split()
                lst.append(float(s_line[0]))

        line = name_of_file.readline()
        # print(line)
        if line != "":
            # print(line)
            s = line.split()
            label = s[2]
        return label, lst

    def forward(self, X, y, index, layer, num_nurons):

        lst = []

        # for i in range(num_nurons):
        w = self.list_of_weights[layer]
        b = self.list_of_biases[layer]
        y_inter = w.dot(X) + b
        y = sigma(y_inter)
        lst.append(y)
        # y = np.array(y).reshape(num_nurons, 1)
        self.list_of_a_i.append(y)
        self.list_of_z_i.append(y_inter)
        # print(y)
        return y

    def main_trainer(self, X, pre_num_nurons, num_of_layers):

        # y = np.random.random((self.initial_data_dim, 1))
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

        # for i in range(num_nurons):
        w = self.list_of_trained["w"][layer]
        b = self.list_of_trained["b"][layer]
        y_inter = w.dot(X) + b
        y = sigma(y_inter)
        lst.append(y)
        # y = np.array(y).reshape(num_nurons, 1)
        self.list_of_a_i.append(y)
        self.list_of_z_i.append(y_inter)
        return y

    def main_trainer_trained(self, X, pre_num_nurons, num_of_layers):

        # y = np.random.random((self.initial_data_dim, 1))
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
            # for i in range(list_of_layers[i]):
            #     Cost = Cost + 2 * (y[l] - self.list_of_z_i)
            # y = self.list_of_z_i[i]

        return Cost

    def modify_w(self, W):
        index = 0
        # index2 =0
        row = 0
        col = 0
        sum_m = 0
        n = 0
        i = 0
        t = self.num_layers
        # print(W)
        d1 = W[0][index].shape[0]
        d2 = W[0][index].shape[1]
        # print(d1)
        # print(d2)

        # some_w = np.array(W[0][index]).reshape((self.list_of_weights[index].shape[0], self.list_of_weights[index].shape[1]))
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
        # lst_w = []
        # lst_b = []
        # new_w = 0
        # new_b = 0
        # for i in range(num_layers, -1, -1):
        #     z = self.list_of_z_i[i]
        #     ai = sigma(self.list_of_z_i[i])
        #     ai_minus = sigma(self.list_of_z_i[i - 1])
        #
        #     delta_w = ai_minus * self.derivative(z) * self.cost_of_ai(i, y)
        #     delta_b = self.derivative(z) * self.cost_of_ai(ai, y)
        #     lst_w.append(delta_w)
        #     lst_b.append(delta_b)
        #     new_w = self.list_of_weights + (np.array(lst_w).reshape(self.list_of_weights[i].shape[0])).dot(self.alpha)
        #     new_b = self.list_of_biases + np.array(lst_b).reshape(self.list_of_biases[i].shape[0]).dot(self.alpha)
        #     self.list_of_weights[i] = new_w
        #     self.list_of_biases[i] = new_b

        y = np.array([1, 0])
        if desired_y == 0:
            com_lst = [1, 0]
        else:
            com_lst = [0, 1]
        p = -2
        # z_i = self.list_of_z_i[p]
        # a_i = sigma(self.list_of_z_i[p])
        # a_t = sigma(self.list_of_z_i[p - 1])
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
            # print(self.list_of_weights)

    def training_procudure(self):
        # produce and X
        # while not the end of file
        new_data = self.read_one_set_of_data(network.open_file_data_all)
        # print(new_data)

        while len(new_data[1]) != 0:

            lst_of_data = new_data[1]
            # print("dada")
            # print(lst_of_data)

            label = new_data[0]
            # print(label)
            if label == "buy":
                desired_y = 0
            else:
                desired_y = 1
            # cost and back
            # go forwards and train
            # y = []
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
                #
                # y = self.main_trainer(lst_of_data, self.initial_data_dim, self.num_layers)
                #
                # # cost and back
                # if desired_y == 0:
                #     c1 = y[0] - 1
                #     c2 = y[1] - 0
                # else:
                #     c1 = y[0] - 0
                #     c2 = y[1] - 1
                # new_y = [c1, c2]

                self.back(np.array(new_y).reshape(2, 1), desired_y)
                y = self.main_trainer(lst_of_data, self.initial_data_dim, self.num_layers)

            L1 = self.list_of_weights
            L2 = self.list_of_biases

            self.list_of_trained_w_and_b.get("w").append(L1)

            self.list_of_trained_w_and_b.get("b").append(L2)
            # print(L1)


            new_data = self.read_one_set_of_data(network.open_file_data_all)

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


# what the user can do


num = 3

list_of_layers = [10, 10, 2]
input_data_count = 24
error = 0.01

file_hourly = "/Users/amir/Desktop/fxtime/cur_hours.csv"
file_daily = "/Users/amir/Desktop/fxtime/cur_daily.csv"
des_file = "/Users/amir/Desktop/fxtime/currency_new_version.csv"

lst_of_new_data = [0.78751, 0.78761, 0.78754, 0.78743, 0.7873, 0.78741, 0.7871,
                   0.78773, 0.78774, 0.78776, 0.78809, 0.78805, 0.78777, 0.78823, 0.78844, 0.78807,
                   0.78814, 0.78814, 0.78817, 0.78834, 0.78824, 0.78821, 0.78825, 0.78833]

#
# file_hourly = "/Users/amir/Desktop/fxtime/curr_hours_mod.csv"
# file_daily = "/Users/amir/Desktop/fxtime/curr_daily_mod.csv"
# des_file = "/Users/amir/Desktop/fxtime/currency_new_version_mod.csv"






nn = network(error, num, list_of_layers, [file_daily, file_hourly, des_file], input_data_count)
# it should print buy or sell
nn.training_procudure()
print(nn.predict(lst_of_new_data))
