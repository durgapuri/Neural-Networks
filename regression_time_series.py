#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[1]:


import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
def collect_data(step_value):
    file_read = open(direct, "r")
    file_lines = file_read.readlines();
#     print(len(file_lines))
    index_list = []
    start_index_list = []
    end_index_list = []
    active_values = []
    for lines in file_lines[1:step_value]:
        line_vals = lines.split(";")
        val = line_vals[2]
        ind = len(active_values)
        if val == "?":
            index_list.append(ind)
        else:
            active_values.append(float(val))
            
    val = np.mean(active_values)
    for i in range(len(index_list)):
        active_values.insert(index_list[i], val)
        print(val)
    
    index_list = []
    for lines in file_lines[step_value:]:
        line_vals = lines.split(";")
        val = line_vals[2]
        ind = len(active_values)
        if val == "?":
            index_list.append(ind)
            if ind-60>=0:
                start_index_list.append(ind-60)
                end_index_list.append(ind-1)
            else:
                start_index_list.append(0)
                end_index_list.append(ind-1)
        else:
            active_values.append(float(val))
    return index_list, start_index_list, end_index_list, active_values

def prepare_data(active_values): 
    train_list = []
    label_list = []
    for i in range(len(active_values)):
        range_b = i
        if i+60 < len(active_values):
            range_e = i+60
        else:
            break
        train_list.append(active_values[range_b:range_e])
        label_list.append(active_values[range_e])
    return train_list, label_list

# direct = "/home/jyoti/Documents/SMAI/assign3/Q4/household_power_consumption.txt"
direct = sys.argv[1]
step_value = 60
index_list, start_index_list, end_index_list, active_values = collect_data(step_value)
# print("index_list",len(index_list))
# print("start_index_list", len(start_index_list))
# print("end_index_list", len(end_index_list))
# print("active_values", len(active_values))
train_list, label_list = prepare_data(active_values)
# print(len(train_list), len(label_list))
train_array = np.array(train_list)
label_array = np.array(label_list)
# print(train_array.shape)
# print(label_array.shape)  
md = Sequential()
md.add(Dense(100, activation="relu", input_dim=step_value))
md.add(Dense(1))
md.compile(optimizer="adam", loss="mse", metrics=['mse'])
md.fit(train_array, label_array, epochs=20, batch_size=500, verbose=0)
# print(len(index_list))
for i in range(len(index_list)):
    x_input = np.array(active_values[start_index_list[i]:end_index_list[i]+1])
    x_input = x_input.reshape((1,step_value))
    y_val = md.predict(x_input, verbose=0)
    active_values.insert(index_list[i], y_val)
    print(y_val[0][0])









