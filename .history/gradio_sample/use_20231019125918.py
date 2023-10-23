from fastai.vision.all import *
import gradio as gr

learn = load_learner('/Users/mac/Desktop/export.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    print("pred: ", pred)
    print("pred_idx: ", pred)
    print("pred: ", pred)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=5)).launch(share=True)