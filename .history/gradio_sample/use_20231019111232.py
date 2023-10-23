from fastai.vision.all import *

learn = load_learner('/Users/mac/Desktop/model.pkl')
labels = learn.dls.vocab
def predict(img):
        img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}