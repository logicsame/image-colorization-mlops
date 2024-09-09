import streamlit as st
from src.imagecolorization.pipeline.prediction import ImageColorizationSystem
from PIL import Image
from io import BytesIO



# Streamlit app
st.title("Image Colorization App")
st.write("Upload a black-and-white image, and this app will colorize it.")

# Load the model
colorization_system = ImageColorizationSystem("C:\\mlops project\\image-colorization-mlops\\artifacts\\trained_model\\cwgan_generator_final.pt", "C:\\mlops project\\image-colorization-mlops\\artifacts\\trained_model\\cwgan_critic_final.pt")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Convert image to grayscale and colorize it
    grayscale_image = colorization_system.load_image(image)
    colorized_image = colorization_system.colorize(grayscale_image)
    
    # Convert to Image and display
    colorized_image_pil = Image.fromarray((colorized_image * 255).astype('uint8'))
    st.image(colorized_image_pil, caption='Colorized Image', use_column_width=True)

    # Option to download the colorized image
    buf = BytesIO()
    colorized_image_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("Download Colorized Image", byte_im, file_name="colorized_image.png")
