import numpy as np
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


with gr.Blocks() as demo:
    gr.Markdown(
        "# Style Transfer <br>")
    gr.Markdown(
        "Choose the method first and then choose the style image and content image!")
    with gr.Tab("Model1"):
        with gr.Row():
            image_input0 = gr.Image(label="Content Image")
            image_input1 = gr.Image(label="Style Image")
            with gr.Column():
                gr.Markdown(
                    "### This models's style must be chosen in these 4 styles:")
                styles = gr.Examples(examples=["styles/Arles.jpg", "styles/starryNight.jpg", "styles/theMuse.jpg", "styles/UnderTheWave.jpg"], inputs=[image_input1], label="Style Image")
        image_button1 = gr.Button("Transfer")
        image_output0 = gr.Image(label="Output Image")
    with gr.Tab("Model2"):
        with gr.Row():
            image_input2 = gr.Image(label="Content Image")
            image_input3 = gr.Image(label="Style Image")
            with gr.Column():
                gr.Markdown(
                    "### This models's style must be chosen in these 4 styles:")
                styles2 = gr.Examples(examples=["styles/Arles.jpg", "styles/starryNight.jpg", "styles/theMuse.jpg", "styles/UnderTheWave.jpg"], inputs=[image_input3], label="Style Image")
        image_button2 = gr.Button("Transfer")
        image_output1 = gr.Image(label="Output Image")
    with gr.Tab("Model3"):
        with gr.Row():
            image_input4 = gr.Image(label="Content Image")
            image_input5 = gr.Image(label="Style Image")
        image_button3 = gr.Button("Transfer")
        image_output2 = gr.Image(label="Output Image")
    with gr.Tab("Model4"):
        with gr.Row():
            image_input6 = gr.Image(label="Content Image")
            image_input7 = gr.Image(label="Style Image")
        image_button4 = gr.Button("Transfer")
        image_output3 = gr.Image(label="Output Image")
    image_button1.click(perform_style_transfer, inputs=[
                        image_input0, image_input1], outputs=image_output0)
    image_button2.click(perform_style_transfer, inputs=[
                        image_input2, image_input3], outputs=image_output1)
    image_button3.click(perform_style_transfer, inputs=[
                        image_input4, image_input5], outputs=image_output2)
    image_button4.click(perform_style_transfer, inputs=[
                        image_input6, image_input7], outputs=image_output3)

demo.launch()
