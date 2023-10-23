from datasets import Dataset

def f(x):
    

dataset = Dataset.from_dict({"a": [0,1,2]})
print(dataset)
dataset.map()