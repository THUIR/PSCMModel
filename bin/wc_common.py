import math

def arr_string(arr, sep="\t"):
    info = ""
    if len(arr) > 0:
        info += str(arr[0])
        for i in range(1, len(arr)):
            info += sep + str(arr[i])
    return info

def string_arr(str, sep, process_function):
    arr = str.strip().split(sep)
    ret = []
    for i in range(0, len(arr)):
        if process_function == "int":
            ret.append(int(arr[i]))
        elif process_function == "float":
            if arr[i] == "":
                ret.append(0.0)
            else:
                ret.append(float(arr[i]))
        else:
            ret.append(arr[i])
    return ret

def matrix_string(arr, inner_sep="\t", outer_sep="\n"):
    info = ""
    for i in range(0, len(arr)):
        info += arr_string(arr[i], inner_sep)
        info += outer_sep
    return info

def arr_string_index(arr, index, sep="\t"):
    info = ""
    if len(arr) > 0:
        info += str(arr[0][index])
        for i in range(1, len(arr)):
            info += sep + str(arr[i][index])
    return info

def matrix_string_index(arr, index, inner_sep="\t", outer_sep="\n"):
    info = ""
    for i in range(0, len(arr)):
        info += arr_string_index(arr[i], index, inner_sep)
        info += outer_sep
    return info

class Rect:
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
    def in_rect(self, x, y):
        return x >= self.x1 and x < self.x2 and y >= self.y1 and y < self.y2
    def to_string(self):
        return str(self.x1) + "," + str(self.x2) + "," + str(self.y1) + "," + str(self.y2)

def init_22():
    ret = []
    ret.append([])
    ret.append([])
    ret[0].append(0)
    ret[0].append(0)
    ret[1].append(0)
    ret[1].append(0)
    return ret

def print_22(title, row, column, matrix):
    print title + "\t" + column + "=0\t" + column + "=1"
    print row + "=0\t" + str(matrix[0][0]) + "\t" + str(matrix[0][1])
    print row + "=1\t" + str(matrix[1][0]) + "\t" + str(matrix[1][1])

def print_22_info(title, row, column, matrix):
    info = ""
    info += title + "\t" + column + "=0\t" + column + "=1" + '\n'
    info += row + "=0\t" + str(matrix[0][0]) + "\t" + str(matrix[0][1]) + '\n'
    info += row + "=1\t" + str(matrix[1][0]) + "\t" + str(matrix[1][1]) + '\n'
    return info

def print_22_trans(title, row, column, matrix):
    print title + "\t" + row + "=0\t" + row + "=1"
    #print column + "=0\t" + str(matrix[0][0]) + "\t" + str(matrix[1][0])
    #print column + "=1\t" + str(matrix[0][1]) + "\t" + str(matrix[1][1])
    print str(matrix[0][0]) + "\t" + str(matrix[1][0])
    print str(matrix[0][1]) + "\t" + str(matrix[1][1])

def set_string(arr, list):
    info = ""
    if len(list) > 0:
        info += str(arr[list[0]])
        for i in range(1, len(arr)):
            info += "\t" + str(arr[list[i]])
    return info        
 
def get_index(name, arr):
    for i in range(0, len(arr)):
        if name == arr[i]:
            return i
    print "Get index error: " + str(name)
    return 0 

def load_valid_user(filename):
    user_set = {}
    user_list = []
    in_file = open(filename)
    in_file.readline()
    line_list = in_file.readlines()
    in_file.close()
    for line in line_list:
        arr = line.strip().split("\t")
        user = arr[0]
        if user != "":
            user_set[user] = arr[1:]
            user_list.append(user)
    return (user_set, user_list)

def load_mouse_feature(filename):
    mouse_feature_set = {}
    in_file = open(filename)
    mouse_feature_name_list = in_file.readline().strip().split("\t")
    line_list = in_file.readlines()
    in_file.close()
    user_index = get_index("user_id", mouse_feature_name_list)
    page_index = get_index("index", mouse_feature_name_list)
    result_index_index = get_index("rank", mouse_feature_name_list)
    for line in line_list:
        arr = line.strip().split("\t")
        if len(arr) == len(mouse_feature_name_list):
            user = arr[user_index]
            index = int(arr[page_index])
            result_index = int(arr[result_index_index])
            if not mouse_feature_set.has_key(user):
                mouse_feature_set[user] = {}
            if not mouse_feature_set[user].has_key(index):
                mouse_feature_set[user][index] = {}
            mouse_feature_set[user][index][result_index] = arr
    return (mouse_feature_set, mouse_feature_name_list)

def load_mouse_feature_arff(filename):
    mouse_feature_set = {}
    in_file = open(filename)
    mouse_feature_name_list = []
    while True:
        line = in_file.readline()
        if line.startswith("@data"):
            break
        if line.startswith("@attribute"):
            arr = line.strip().split(" ")
            mouse_feature_name_list.append(arr[1])
    line_list = in_file.readlines()
    in_file.close()
    user_index = get_index("user_id", mouse_feature_name_list)
    page_index = get_index("index", mouse_feature_name_list)
    result_index_index = get_index("rank", mouse_feature_name_list)
    for line in line_list:
        arr = line.strip().split(",")
        if len(arr) == len(mouse_feature_name_list):
            user = arr[user_index]
            index = int(arr[page_index])
            result_index = int(arr[result_index_index])
            if not mouse_feature_set.has_key(user):
                mouse_feature_set[user] = {}
            if not mouse_feature_set[user].has_key(index):
                mouse_feature_set[user][index] = {}
            mouse_feature_set[user][index][result_index] = arr
    return (mouse_feature_set, mouse_feature_name_list)
    
def load_result_coordinate(result_coordinate_file_name):
    #load each result's area
    index_coordinate_set = {}
    index_result_num_set = {}
    result_coordinate_file = open(result_coordinate_file_name)
    result_coordinate_file.readline()
    result_coordinate_file.readline()
    while True:
        line = result_coordinate_file.readline()
        if not line:
            break
        list = line.strip().split("\t")
        if len(list) != 2:
            continue
        index = int(list[0])
        result_num = int(list[1])
        if not index_coordinate_set.has_key(index):
            index_coordinate_set[index] = []
        index_result_num_set[index] = result_num
        for i in range(0, result_num):
            rect_list = result_coordinate_file.readline().strip().split("\t")
            result_rect = Rect(int(rect_list[0]), int(rect_list[1]) , int(rect_list[0]) + int(rect_list[2]), int(rect_list[1]) + int(rect_list[3]))
            index_coordinate_set[index].append(result_rect)
    result_coordinate_file.close()
    return (index_coordinate_set, index_result_num_set)

def load_human_relevance_binary_label(filename):#median label
    relevance_set = {}
    in_file = open(filename)
    info_list = in_file.readline().strip().split("\t")
    people_num = len(info_list) - 2
    line_list = in_file.readlines()
    in_file.close()
    for line in line_list:
        arr = line.strip().split("\t")
        if len(arr) != len(info_list):
            continue
        index = int(arr[0])
        result_index = int(arr[1])
        label_list = []
        for i in range(0, people_num):
            label_list.append(int(arr[2 + i]))
        label_list = sorted(label_list, key=lambda x : x, reverse=False)
        final_label = label_list[people_num / 2]
        if final_label > 2:
            final_label = 1
        else:
            final_label = 0
        if not relevance_set.has_key(index):
            relevance_set[index] = {}
        relevance_set[index][result_index] = final_label
    return relevance_set
        
        
#compare two examine
#eyetracking data is real value
#check data is predict value
#              eyetracking
#            p(examine)           n(not examine)
# check  p'    check_p_eye_p(TP)    check_p_eye_n(FP)
#        n'    check_n_eye_p(FN)    check_n_eye_n(TN)
def compute_ROC(check_p_eye_p, check_p_eye_n, check_n_eye_p, check_n_eye_n):
    TP = float(check_p_eye_p)
    TN = float(check_n_eye_n)
    FP = float(check_p_eye_n)
    FN = float(check_n_eye_p)
    P = TP + FN
    N = TN + FP
    accuracy = (TP + TN) / (P + N)
    precision = (TP) / (TP + FP)
    recall = (TP) / P
    F = (2.0 * precision * recall) / (precision + recall)
    MCC = (TP * TN - FP * FN) / (math.sqrt(TP +FP) * math.sqrt(TP +FN) * math.sqrt(TN +FP) * math.sqrt(TN +FN))
    PRa = (TP + TN) / (P + N)
    PRe = ((TP + FP) / (P + N)) * ((TP + FN) / (P + N)) + ((FP + TN) / (P + N)) * ((FN + TN) / (P + N))
    Kappa = (PRa - PRe) / (1.0 - PRe)
    ret = ""
    ret  = ret + "--ROC--" + "\n"
    ret  = ret + "\t\teyetracking" + "\n"
    ret  = ret + "\t\tp(examine)\tn(not examine)" + "\n"
    ret  = ret + "check\tp'\t" + str(check_p_eye_p) + "\t" + str(check_p_eye_n) + "\n"
    ret  = ret + "\tn'\t" + str(check_n_eye_p) + "\t" + str(check_n_eye_n) + "\n"
    ret  = ret + "accuracy\t" + str(accuracy) + "\n"
    ret  = ret + "precision\t" + str(precision) + "\n"
    ret  = ret + "recall\t" + str(recall) + "\n"
    ret  = ret + "F Measure\t" + str(F) + "\n"
    ret  = ret + "MCC\t" + str(MCC) + "\n"
    ret  = ret + "Kappa\t" + str(Kappa) + "\n"
    return ret
    
def compute_ALL(TN, FP, FN, TP):
    TP = float(TP)
    TN = float(TN)
    FP = float(FP)
    FN = float(FN)
    P = TP + FN
    N = TN + FP
    FPR = 0.0
    if FP + TN > 0:
        FPR = FP / (FP + TN)
    accuracy = 0.0
    if (P + N) > 0:
        accuracy = (TP + TN) / (P + N)
    precision = 0
    if TP + FP > 0:
        precision = (TP) / (TP + FP)
    recall = 0
    if P > 0:
        recall = (TP) / P
    F = 0.0
    if precision + recall > 0:
        F = (2.0 * precision * recall) / (precision + recall)
    Kappa = 0.0
    MCC = 0.0
    if P + N > 0:
        #MCC = (TP * TN - FP * FN) / (math.sqrt(TP +FP) * math.sqrt(TP +FN) * math.sqrt(TN +FP) * math.sqrt(TN +FN))
        PRa = (TP + TN) / (P + N)
        PRe = ((TP + FP) / (P + N)) * ((TP + FN) / (P + N)) + ((FP + TN) / (P + N)) * ((FN + TN) / (P + N))
        if abs(1.0 - PRe) > 0:
            Kappa = (PRa - PRe) / (1.0 - PRe)
    return (accuracy, precision, recall, F, Kappa, FPR)
    
def compute_avg_var(list):
    avg = 0.0
    var = 0.0
    N = len(list)
    if N > 0:
        for i in range(0, N):
            avg += list[i]
        avg /= N
        for i in range(0, N):
            var += (list[i] - avg) * (list[i] - avg)
        var /= N
    return (avg, var)

def compute_correlation(x, y, max_len):
    length = max_len
    if len(x) < length:
        length = len(x)
    if len(y) < length:
        length = len(y)
    if length == 0:
        return 0.0
    x_avg = 0.0
    y_avg = 0.0
    for i in range(0, length):
        x_avg = x_avg + x[i]
        y_avg = y_avg + y[i]
    x_avg = x_avg / length
    y_avg = y_avg / length
    a = 0.0
    b = 0.0
    c = 0.0
    for i in range(0, length):
        a = a + (x[i] - x_avg) * (y[i] - y_avg)
        b = b + (x[i] - x_avg) * (x[i] - x_avg)
        c = c + (y[i] - y_avg) * (y[i] - y_avg)
    if b <= 0 or c <= 0:
        # print "X = " + str(x)
        # print "Y = " + str(y)
        return 0.0
    return (a / math.sqrt(b * c))

def load_arff_line(file_name):
    in_file = open(file_name)
    line_list = in_file.readlines()
    in_file.close()
    head_list = []
    data_list = []
    head_flag = 1
    for line in line_list:
        if line.startswith("@data"):
            head_flag = 0
            continue
        if head_flag == 1:
            head_list.append(line)
        else:
            data_list.append(line)
    return (head_list, data_list)