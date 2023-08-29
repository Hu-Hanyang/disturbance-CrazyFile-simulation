import os

absolute_path = os.path.dirname(__file__)
# print(absolute_path)
# print(os.getcwd())
relative_path = "train_results/"
full_path = os.path.join(absolute_path, relative_path)
# print(full_path)