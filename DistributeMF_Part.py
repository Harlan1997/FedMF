import time
import copy
import numpy as np
import sys
import os
import config_file as cfg_file
import sys
import datetime


from load_data import ratings_dict, item_id_list, user_id_list, test_data
from shared_parameter import *


def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
    fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d1')
    sys.stdout = Logger(fileName + '.log', path=path)
    print(fileName.center(60,'*'))


def user_update(single_user_vector, user_rating_list, item_vector):
    gradient = {}
    for item_id, rate, _ in user_rating_list:
        error = rate - np.dot(single_user_vector, item_vector[item_id])
        single_user_vector = single_user_vector - lr * (-2 * error * item_vector[item_id] + 2 * reg_u * single_user_vector)
        gradient[item_id] = error * single_user_vector
    return single_user_vector, gradient


def mse():
    loss = []
    for i in range(len(user_id_list)):
        for r in range(len(ratings_dict[user_id_list[i]])):
            item_id, rate, _ = ratings_dict[user_id_list[i]][r]
            error = (rate - np.dot(user_vector[i], item_vector[item_id])) ** 2
            loss.append(error)
    return np.mean(loss)


if __name__ == '__main__':
    
    # Init process
    user_vector = np.random.normal(size=[len(user_id_list), hidden_dim])
    item_vector = np.random.normal(size=[len(item_id_list), hidden_dim])

    start_time = time.time()
    make_print_to_file()
    for iteration in range(max_iteration):

        print('###################')
        t = time.time()

        # Step 2 User updates
        gradient_from_user = []
        for i in range(len(user_id_list)):
            user_vector[i], gradient = user_update(user_vector[i], ratings_dict[user_id_list[i]], item_vector)
            gradient_from_user.append(gradient)

        # Step 3 Server update
        tmp_item_vector = copy.deepcopy(item_vector)
        for g in gradient_from_user:
            for item_id in g:
                item_vector[item_id] = item_vector[item_id] - lr * (-2 * g[item_id] + 2 * reg_v * item_vector[item_id])
        e = np.mean(np.abs(item_vector - tmp_item_vector))
        
        if e < 1e-4:
            print('Converged')
            break
        print('mean change of item vector', e)
        print('iteration', iteration)
        print('Time', time.time() - t, 's')
        print('loss', mse())
    print('Converged using', time.time() - start_time)

    prediction = []
    real_label = []

    # testing
    for i in range(len(user_id_list)):

        p = np.dot(user_vector[i:i+1], np.transpose(item_vector))[0]

        r = test_data[user_id_list[i]]

        real_label.append([e[1] for e in r])
        prediction.append([p[e[0]] for e in r])

    prediction = np.array(prediction, dtype=np.float32)
    real_label = np.array(real_label, dtype=np.float32)

    print('rmse', np.sqrt(np.mean(np.square(real_label - prediction))))