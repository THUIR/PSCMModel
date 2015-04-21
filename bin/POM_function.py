import sys, os, re, urllib, math, random , safe_math

class Path:
    prob = 0.0
    v_list = [] # list of visit element index
    s_list = [] # list of 0/1 ,skip
    def __init__(self, prob, v_list, s_list):
        self.prob = prob
        self.v_list = v_list
        self.s_list = s_list
    def tostring(self):
        ret = str(self.prob)
        for i in range(0, len(self.v_list)):
            ret = ret + "\t" + str(self.v_list[i]) + "(" + str(self.s_list[i]) + ")"
        return ret
        
def compute_P_given_A(path, param_V, param_S, param_first_click, first_flag, position_limit):
    prob = 0.0
    if first_flag == 0 :
        prob = 1.0
    elif len(path.s_list) == 0:
        prob = 0.0
    else:
        prob = param_first_click[path.v_list[0]]
        if path.s_list[0] == 1:
            if not path.v_list[0] == position_limit:
                prob = prob * param_S[path.v_list[0]]
        else:
            if not path.v_list[0] == position_limit:
                prob = prob * (1.0 - param_S[path.v_list[0]])
        for i in range(1, len(path.s_list)):
            previous_v = path.v_list[i - 1]
            current_v = path.v_list[i]
            current_s = path.s_list[i]
            prob = prob * param_V[previous_v][current_v]
            if current_s == 1:
                if not current_v == position_limit:
                    prob = prob * param_S[current_v]
            else:
                if not current_v == position_limit:
                    prob = prob * (1.0 - param_S[current_v])
    path.prob = prob
    return prob

def copy_path(v_list, s_list):
    new_v_list = []
    new_s_list = []
    for j in range(0, len(v_list)):
        new_v_list.append(v_list[j])
        new_s_list.append(s_list[j])
    path = Path(0.0, new_v_list, new_s_list)
    return path

def insert_path_list(list, p, max_node_num):
    insert_flag = True
    insert_pos = len(list)
    for i in range(0, len(list)):
        if p.prob < list[len(list)- 1 - i].prob:
            insert_pos = len(list) - i
            break
    if insert_pos == max_node_num:
        insert_flag = False
    if insert_flag == True:
        list.insert(insert_pos, p)
        for i in range(max_node_num, len(list)):
            list.pop()
    return insert_flag
    
def add_Qk_list(insert_interval, interval_num, path_list, v_list, s_list, param_V, param_S, param_first_click, first_flag, max_qk_length, max_insert_num, position_limit, max_node_num, basic_flag):
    if len(s_list) == 0:
        return
    in_sert_flag = True
    if basic_flag == 1:
        basic_path = copy_path(v_list, s_list)
        compute_P_given_A(basic_path, param_V, param_S, param_first_click, first_flag, position_limit)
        in_sert_flag = insert_path_list(path_list, basic_path, max_node_num)
    if in_sert_flag == False:
        return
    if max_insert_num <= 0:
        return
    insert_begin = 0
    insert_end = 0
    interval_index = 0
    insert_flag = 0
    for i in range(1, len(s_list)):
        if s_list[i] == 0:
            insert_begin = insert_end
            insert_end = i
            if interval_index == insert_interval:
                insert_flag = 1
                break
            interval_index = interval_index + 1
    if insert_flag == 0:
        return
    #print str(insert_begin) + " - " + str(insert_end)
    # for j in range(insert_begin, insert_end + 1):
        # print v_list[j]
    for i in range(0, position_limit):
        occur = 0
        for j in range(insert_begin, insert_end + 1):
            if i == v_list[j]:
                occur = 1
        if occur == 1 or insert_end - insert_begin + 2 > max_qk_length:
            continue
        v_list.insert(insert_end, i)
        s_list.insert(insert_end, 1)
        #print path.tostring()
        add_Qk_list(insert_interval, interval_num, path_list, v_list, s_list, param_V, param_S, param_first_click, first_flag, max_qk_length, max_insert_num -1, position_limit, max_node_num, 1)
        del v_list[insert_end]
        del s_list[insert_end]
    add_Qk_list(insert_interval + 1, interval_num, path_list, v_list, s_list, param_V, param_S, param_first_click, first_flag, max_qk_length, max_insert_num, position_limit, max_node_num, 0)
