from fastai.vision.all import *
import gradio as gr

learn = load_learner('/Users/mac/Desktop/export.pkl')
labels = learn.dls.vocab
print(labels)
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    print("pred: ", pred)
    print("pred_idx: ", pred_idx)
    print("probs: ", probs)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
interpretation='default'

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,interpretation=interpretation,enable_queue=enable_queue).launch(share=True)