import math
import random
import numpy as np
import os


def sigma(x):
    return 1 / (1 + np.exp(-x))


def reader_hourly(line, file_write):
    """
    :param line: the line to be processed and interpreted.
    :type line: String
    :param file_write: file to write to
    :type file_write: open File
    :return: tupple of values recveied in the string line.
    :rtype: tupple of integer and string
    """
    splitted = line.split()
    open_price = splitted[2]
    high_price = splitted[3]
    low_price = splitted[4]
    close_price = splitted[5]
    if open_price <= close_price:
        result = (round(sigma(float(open_price)), 5), "buy")
    else:
        result = (round(sigma(float(open_price)), 5), "sell")
    # file_write.write(str(result[0]) + " " + str(result[1]) + "\n")
    str1 = str(result[0]) + " " + str(result[1]) + "\n"
    file_write.flush()
    # os.fsync(file_write)

    return str1


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
        str2 = "OpenDaily" + " " + str(result[0]) + " " + str(result[1]) + "\n"
        # write_to_file.flush()
        # os.fsync(write_to_file)
        # print(result)
    return str2


def provide_data(daily, hourly, file_write):
    str1 = ""
    str2 = ""
    file_read_h = open(hourly)
    file_read_d = open(daily)
    open_file = open(file_write, "a")
    line = file_read_d.readline()
    line = file_read_h.readline()
    # line = file_read_hourly.readline()
    lst = []
    # label = ""
    while line != "":
        for i in range(24):
            line = file_read_h.readline()
            if line != "":
                # print(line + "this is form hours")
                str1 = str1 + reader_hourly(line, open_file)
                # open_file.write(str1)
                # open_file.flush()
                # os.fsync(open_file.fileno())

                # lst.append(float(res[0]))
        line = file_read_d.readline()
        # print("print of line daily is .........." + line + "\n")
        if len(line) != 0:
            str2 = reader_daily(line, open_file)
            print(str2)
        open_file.write(str2 + str1)
        open_file.flush()
        os.fsync(open_file.fileno())
        str1 = ""
        str2 = ""

    open_file.close()
