import math
import random
import numpy as np
import os


def sigma(x):
    return 1 / (1 + np.exp(-x))



def reader_daily(line, write_to_file):
    """
    :param line: the line to be processed and interpreted.
    :type line: String
    :param write_to_file: file to write to
    :type write_to_file: open File
    :return: tupple of values recveied in the string line
    :rtype: tupple of integer and string
    """

    # print(line)
    str2 = ""
    splitted = line.split()
    # print(splitted)
    open_price = splitted[1]
    high_price = splitted[2]
    low_price = splitted[3]
    close_price = splitted[4]
    if open_price <= close_price:
        result = (round(sigma(float(open_price)), 5), "buy")
    else:
        result = (round(sigma(float(open_price)), 5), "sell")
        # write_to_file.write("OpenDaily" + " " + str(result[0]) + " " + str(result[1]) + "\n")
        str2 = str(result[1])
        # write_to_file.flush()
        # os.fsync(write_to_file)
        # print(result)
    return str2


# buy 1 and sell 0

def provide_output(daily, file_write):

    file_read_d = open(daily)
    open_file = open(file_write, "a")
    line = file_read_d.readline()
    line = file_read_d.readline()
    line = file_read_d.readline()

    lst = []
    # label = ""
    c = 1
    while c <= 50:
        # line = file_read_h.readline()
        if line != "":
            splitted = line.split()
            # print(splitted)
            open_price = splitted[1]
            high_price = splitted[2]
            low_price = splitted[3]
            close_price = splitted[4]
            if open_price <= close_price:
                result = (round(sigma(float(open_price)), 5), 0)
            else:
                result = (round(sigma(float(open_price)), 5), 1)
            lst.append(result[1])
            # print(c, float(open_price), str(result[1]))
            open_file.write(str(result[1]) + ",")
        line = file_read_d.readline()
        c = c+1

    open_file.close()
    return lst




def provide_data(input_file, file_write):

    file_read_d = open(input_file)
    open_file = open(file_write, "a")
    line = file_read_d.readline()
    line =file_read_d.readline()
    # line = file_read_d.readline()

    lst = []

    c= 1
    # label = ""
    while c <= 1200:
        # line = file_read_h.readline()
        if line != "":
            splitted = line.split()
            # print(splitted)
            open_price = splitted[2]
            high_price = splitted[3]
            low_price = splitted[4]
            close_price = splitted[5]
            if open_price <= close_price:
                result = (round(sigma(float(open_price)), 5), "buy")
            else:
                result = (round(sigma(float(open_price)), 5), "sell")
            # print(c ,result[0])
            lst.append(result[0])
            open_file.write(str(result[0]) + ", ")
                # print(c,float(open_price), str(result[0]))
        line = file_read_d.readline()

        c= c +1

                # open_file.flush()

    open_file.close()
    print(len(lst))
    return lst



