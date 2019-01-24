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
                result = (round(float(open_price), 5), 0)
            else:
                result = (round(float(open_price), 5), 1)
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
    training_lst = []
    c = 1
    while c <= 1200:
        if line != "":
            splitted = line.split()
            open_price = splitted[2]
            high_price = splitted[3]
            low_price = splitted[4]
            close_price = splitted[5]
            if open_price <= close_price:
                result = (round(float(open_price), 5), "buy")
            else:
                result = (round(float(open_price), 5), "sell")
            training_lst.append(result[0])
            open_file.write(str(result[0]) + ", ")
        line = file_read_d.readline()
        c = c + 1
    open_file.close()
    return training_lst






def provide_test(test_file):

    file_read_d = open(test_file)
    line = file_read_d.readline()
    line = file_read_d.readline()
    line = file_read_d.readline()
    lst = []

    if line != "":
        splitted = line.split()
        open_price = splitted[1]
        high_price = splitted[2]
        low_price = splitted[3]
        close_price = splitted[4]
        if open_price <= close_price:
            result = (round(float(open_price), 5), 0)
        else:
            result = (round(float(open_price), 5), 1)
        lst.append(result[1])
    line = file_read_d.readline()

    return lst
