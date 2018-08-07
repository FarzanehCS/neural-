
from sklearn.neural_network import MLPClassifier
import data_skikit
import numpy as np

file_hourly = "/Users/amir/Desktop/fxtime/cur_hours.csv"
file_daily = "/Users/amir/Desktop/fxtime/cur_daily.csv"
data_file = "/Users/amir/Desktop/fxtime/input.csv"
out_file = "/Users/amir/Desktop/fxtime/out.csv"

lst_of_new_data = [0.78751, 0.78761, 0.78754, 0.78743, 0.7873, 0.78741, 0.7871,
                   0.78773, 0.78774, 0.78776, 0.78809, 0.78805, 0.78777, 0.78823, 0.78844, 0.78807,
                   0.78814, 0.78814, 0.78817, 0.78834, 0.78824, 0.78821, 0.78825, 0.78833]
num_features = 24
num_samples = 50


new_lst = np.array(lst_of_new_data).reshape(1, num_features)






lst_input = data_skikit.provide_data(file_hourly, data_file)

matrix_input = np.array(lst_input).reshape(num_samples, num_features)

print(matrix_input)

lst_labels = data_skikit.provide_output(file_daily, out_file)

matrix_output = np.array(lst_labels).reshape(num_samples,)
print(matrix_output)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
               hidden_layer_sizes=(10, 10, 2), random_state=1)

clf.fit(matrix_input, matrix_output)




result = clf.predict(new_lst)

if result == [1]:
    print("sell")
else:
    print("buy")



