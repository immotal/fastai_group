from datasets import Dataset

def f(x):
    return x["a"] * 2

dataset = Dataset.from_dict({"a": [0,1,2]})
print(dataset)
dataset.map(f, ba)