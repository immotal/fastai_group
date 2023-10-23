from fastai.vision.all import *

learn = load_learner('/Users/mac/Desktop/model.pkl')
labels = learn.dls.vocab
def predict(img):