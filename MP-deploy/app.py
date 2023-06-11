import tensorflow as tf
import gradio as gr
import numpy as np

model = tf.keras.models.load_model('model/Model.h5')

def predict(inp):
  input_arr = tf.keras.preprocessing.image.img_to_array(inp)
  input_arr = np.array([input_arr]) 
  prediction = model.predict(input_arr)
  return (1-prediction)*100

gr.Interface(fn=predict, 
             inputs=gr.Image(shape=(224, 224)),
             outputs=gr.Number()).launch(share=True,server_name="0.0.0.0")



