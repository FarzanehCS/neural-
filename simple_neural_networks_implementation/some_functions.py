import math
import random

import random
import numpy as np


# from_file = "/Users/amir/Desktop/fxtime/currency.cvs"

# file_hourly = "/Users/amir/Desktop/fxtime/cur_hours.csv"
# file_daily = "/Users/amir/Desktop/fxtime/cur_daily.csv"
# des_file = "/Users/amir/Desktop/fxtime/currency_new_version.csv"
#
# file_read_daily = open(file_daily)
# file_read_hourly = open(file_hourly)
# open_file_write = open(des_file, "a")
#




def sigma(x):
    return 1 / (1 + math.exp(-x))




def comma(daily, write_to_h):
    daily_r = open(daily)

    write_w = open(write_to_h, "a")

    # line = daily_r.readline()

    line = daily_r.readline()

    while line != "":
        s = line.split()
        mod_line = ",".join(s)
        write_w.write(mod_line + "\n")
        line = daily_r.readline()
    daily_r.close()
    write_w.close()


def reader_hourly(line, file_write):

    splitted = line.split(sep="\t")
    open_price = splitted[2]
    high_price = splitted[3]
    low_price = splitted[4]
    close_price = splitted[5]

    if open_price <= close_price:
        result = (round(sigma(float(open_price)), 5), "buy")
    else:
        result = (round(sigma(float(open_price)), 5), "sell")

    file_write.write(str(result[0]) + " ")
    file_write.write(str(result[1]) + "\n")

    return result


def file_reader_hourly(file_to_read, write_to_file):
    file_read = open(file_to_read)
    file_write = open(write_to_file, "a")
    line = file_read.readline()

    while line != "":
        splitted = line.split(sep="\t")
        open_price = splitted[3]
        high_price = splitted[4]
        low_price = splitted[5]
        close_price = splitted[6]

        if open_price <= close_price:
            result = (sigma(float(open_price)), "buy")
        else:
            result = (sigma(float(open_price)), "sell")

        file_write.write(str(result[0]) + " ")
        file_write.write(str(result[1]) + "\n")


def file_reader_daily(file_to_read, write_to_file):
    file_read = open(file_to_read)
    open_file_to_write = open(write_to_file, "a")
    line = file_read.readline()
    line = file_read.readline()

    while line != "":
        splitted = line.split(sep="\t")
        open_price = splitted[2]
        high_price = splitted[3]
        low_price = splitted[4]
        close_price = splitted[5]

        if open_price <= close_price:
            result = (round(sigma(float(open_price)), 6), "buy")
        else:
            result = (round(sigma(float(open_price)), 6), "sell")

        open_file_to_write.write(str(result[0]) + " ")
        open_file_to_write.write(str(result[1]) + "\n")


# test

# file_daily = "/Users/amir/Desktop/fxtime/cur_daily.csv"
# file_daily = "/Users/amir/Desktop/fxtime/cur_hours.csv"


file_daily_write = "/Users/amir/Desktop/fxtime/hour.csv"
file_hours_write = "/Users/amir/Desktop/fxtime/day.csv"

#
# file_reader_daily(file_daily, file_daily_write)
# file_reader_hourly(file_hourly, file_hours_write)


# provide_data()


file_hourly = "/Users/amir/Desktop/fxtime/cur_hours.csv"
file_daily = "/Users/amir/Desktop/fxtime/cur_daily.csv"
des_file1 = "/Users/amir/Desktop/fxtime/curr_hours_mod.csv"
des_file2 = "/Users/amir/Desktop/fxtime/curr_daily_mod.csv"

comma(file_daily, des_file2)
comma(file_hourly, des_file1)


def reader_daily(line, write_to_file):
    """
    :param line: the line to be processed and interpreted.
    :type line: String
    :param write_to_file: file to write to
    :type write_to_file: open File
    :return: tupple of values recveied in the string line
    :rtype: tupple of integer and string
    """

    str2 = ""
    splitted = line.split()
    open_price = splitted[1]
    high_price = splitted[2]
    low_price = splitted[3]
    close_price = splitted[4]
    if open_price <= close_price:
        result = (round(sigma(float(open_price)), 5), "buy")
    else:
        result = (round(sigma(float(open_price)), 5), "sell")
        
        str2 = str(result[1])
     
    return str2


# buy 1 and sell 0

def provide_output(daily, file_write):
    file_read_d = open(daily)
    open_file = open(file_write, "a")
    line = file_read_d.readline()
    line = file_read_d.readline()
    line = file_read_d.readline()

    lst = []
   
    c = 1
    while c <= 50:
        if line != "":
            splitted = line.split()
            open_price = splitted[1]
            high_price = splitted[2]
            low_price = splitted[3]
            close_price = splitted[4]
            if open_price <= close_price:
                result = (round(sigma(float(open_price)), 5), 0)
            else:
                result = (round(sigma(float(open_price)), 5), 1)
            lst.append(result[1])
            open_file.write(str(result[1]) + ",")
        line = file_read_d.readline()
        c = c + 1

    open_file.close()
    return lst


def provide_data(input_file, file_write):
    file_read_d = open(input_file)
    open_file = open(file_write, "a")
    line = file_read_d.readline()
    line = file_read_d.readline()

    lst = []

    c = 1
    while c <= 1200:
        if line != "":
            splitted = line.split()
            open_price = splitted[2]
            high_price = splitted[3]
            low_price = splitted[4]
            close_price = splitted[5]
            if open_price <= close_price:
                result = (round(sigma(float(open_price)), 5), "buy")
            else:
                result = (round(sigma(float(open_price)), 5), "sell")
            lst.append(result[0])
            open_file.write(str(result[0]) + ", ")
        line = file_read_d.readline()

        c = c + 1


    open_file.close()
    print(len(lst))
    return lst
