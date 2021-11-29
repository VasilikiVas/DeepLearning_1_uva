import json
import os
from texttable import Texttable

total_dict = {}
for file in os.listdir("./results"):
   sum_er_dict = {}
   with open("./results/"+file) as json_file:
      data = json.load(json_file)
      #init dictionary
      for key, value in data.items():
         key_sum = key.split(" ")[0]
         sum_er_dict[key_sum] = 0
      #caculate sums
      for key, value in data.items():
         key_sum = key.split(" ")[0]
         error = 1 - value 
         sum_er_dict[key_sum] += error
   total_dict[str(file).split(".json")[0]] = sum_er_dict

t = Texttable()
row = ["Model_name / Augmentation", "DenseNet", "ResNet18", "VGG11", "ResNet34", "VGG_bn"]
t.add_row(row)
result_dict = {}
for key, value in total_dict.items():
   for key_in, value_in in value.items():
      if key != "Clean_results":
         CE = value_in / total_dict["ResNet18_results"][key_in]
         if result_dict.get(key_in) == None: 
            result_dict[key_in] = [CE]
         else: 
            result_dict[key_in].append(CE)

for key, value in result_dict.items():
   if key != "clean":
      value.insert(0,key)
      t.add_row(value)
print("+-----------------------------------------------------------------------------+")
print("|                                      CE                                     |")                                  
print(t.draw())
print("\n")

#____________________________________________Calculate RCE___________________________________

total_dict = {}
for file in os.listdir("./results"):
   sum_er_dict = {}
   with open("./results/"+file) as json_file:
      data = json.load(json_file)
      #init dictionary
      for key, value in data.items():
         key_sum = key.split(" ")[0]
         sum_er_dict[key_sum] = 0
      #caculate sums
      for key, value in data.items():
         key_sum = key.split(" ")[0]
         error = 1 - value 
         sum_er_dict[key_sum] += error
   total_dict[str(file).split(".json")[0]] = sum_er_dict

t = Texttable()
row = ["Model_name / Augmentation", "DenseNet", "ResNet18", "VGG11", "ResNet34", "VGG_bn"]
t.add_row(row)
result_dict = {}
for key, value in total_dict.items():
   for key_in, value_in in value.items():
      if key != "Clean_results" and key_in != "clean":
         RCE = (value_in - value["clean"]) / (total_dict["ResNet18_results"][key_in] - total_dict["ResNet18_results"]["clean"])
         if result_dict.get(key_in) == None: 
            result_dict[key_in] = [RCE]
         else: 
            result_dict[key_in].append(RCE)

for key, value in result_dict.items():
   if key != "clean":
      value.insert(0,key)
      t.add_row(value)
print("+-----------------------------------------------------------------------------+")
print("|                                     RCE                                     |")                                  
print(t.draw())
