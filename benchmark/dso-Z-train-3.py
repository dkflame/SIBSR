from dso import DeepSymbolicOptimizer
from dso import DeepSymbolicRegressor
import pandas as pd
import numpy as np
import os
import multiprocessing
import commentjson as json
from copy import deepcopy
import random


# Possible library elements that sympy capitalizes
capital = ["add", "mul", "pow"]

class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):
        self.val = val
        self.children = []

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val # Avoids unnecessary parantheses, e.g. x1()
        return "{}({})".format(self.val, children_repr)


def build_tree(exp_list, node_list):
   """Recursively builds tree from pre-order traversal"""
   op = exp_list.pop(0)
   if op in ["add", "sub", "mul", "div"]:
      n_children=2
   elif op in ["exp","log","sqrt","n2","neg","sin","cos"]:
      n_children=1
   else:
      n_children=0

   # val = repr(op)
   val = op
   # if val in capital:
   #  val = val.capitalize()

   node = Node(val)
   node_list.append(node)

   for _ in range(n_children):
      node.children.append(build_tree(exp_list, node_list))

   return node


def pre_order_traversal(node, result):
   if node:
      result.append(node.val)
      if node.children:
         if (len(node.children)==2):
            child1 = node.children[0]
            child2 = node.children[1]
            pre_order_traversal(child1, result)
            pre_order_traversal(child2, result)
         else:
            child = node.children[0]
            pre_order_traversal(child, result)

   return result


def main():
   # Create and train the model
   config_json = os.path.join("../config/", "benchmark_regression.json")
   new_config_json = os.path.join("../config/benchmark_regression-Z-train-3.json")

   benchmark = "Z-train-3"
   ground_truth = "div,mul,sin,1.0,div,1.0,x1,add,sub,1.0,mul,2.0,mul,cos,1.0,div,1.0,x1,div,1.0,n2,x1"
   ground_truth_list = ground_truth.split(",")
   ground_truth_length = len(ground_truth_list)
   node_list = []
   Ho_lists = []
   node_const = Node("1.0")
   ground_truth_tree = ground_truth_list.copy()
   ground_truth_tree = build_tree(ground_truth_tree, node_list)
   print("========== ground truth list and tree ==========")
   print(ground_truth_list)
   print(ground_truth_tree.__repr__())

   mutate_nums = random.randint(1,ground_truth_length-1)

   mutate_places = random.sample(range(0,ground_truth_length),mutate_nums)
   for mutate_place in mutate_places:
      # print(mutate_place)
      if len(node_list[mutate_place].children) == 2:
         binary_mutate_num = random.randint(0,2)
         if binary_mutate_num == 2:
            node_list[mutate_place].children = [node_const, node_const]
         else:
            node_list[mutate_place].children[binary_mutate_num] = node_const
      elif len(node_list[mutate_place].children) == 1:
         node_list[mutate_place].children  = [node_const]
   
   Ho_tree = ground_truth_tree
   Ho_traversal = []
   Ho_traversal = pre_order_traversal(Ho_tree, Ho_traversal)
   print("========== Ho tree and traversal ==========")
   print(Ho_tree.__repr__())
   print(Ho_traversal)


   # for node in node_list:
   #  if node.children:
   #     Ho_tree = node
   #     Ho_traversal = []
   #     Ho_traversal = pre_order_traversal(Ho_tree, Ho_traversal)
   #     Ho_lists.append(Ho_traversal)
   # expression_traversal = []
   # expression_traversal = pre_order_traversal(ground_truth_tree, expression_traversal)


   with open(config_json, encoding='utf-8') as f:
      json_data = json.load(f)
      json_data["task"]["dataset"]["name"] = benchmark
      json_data["training"]["Ho"] = Ho_traversal

   with open(new_config_json, 'w') as f:
       f.write(json.dumps(json_data))

   try:
      model = DeepSymbolicOptimizer(new_config_json)
      model.train()
   except:
       print("Invalid Ho: ")
       print(Ho_traversal)

if __name__ == "__main__":
   for i in range(100):
      main()

