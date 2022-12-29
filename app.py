import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import os

# load model
style_transfer_model = hub.load(
    "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# function to style transfer 
# from tensorflow_hub
def perform_style_transfer(content_image, style_image):

    content_image = tf.convert_to_tensor(content_image, np.float32)[
        tf.newaxis, ...] / 255.
    style_image = tf.convert_to_tensor(style_image, np.float32)[
        tf.newaxis, ...] / 255.

    output = style_transfer_model(content_image, style_image)
    stylized_image = output[0]

    return Image.fromarray(np.uint8(stylized_image[0] * 255))

# input content image and style image 
content_image_input = gr.inputs.Image(label="Content Image")
style_image_input = gr.inputs.Image(shape=(256, 256), label="Style Image")


# interface
app_interface = gr.Interface(fn=perform_style_transfer,
                             inputs=[content_image_input, style_image_input],
                             outputs="image",
                             title="Style Transfer",
                             description="Using Gradio and a pretrained Image Stylization model from TensorFlow Hub for online Fast Neural Style Transfer. Simply upload a content image and a style image."
                            )
# launch
# app_interface.launch(share=True)
app_interface.launch()

