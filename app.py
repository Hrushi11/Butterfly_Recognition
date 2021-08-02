import tensorflow as tf
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

IMAGE_SHAPE = (224, 224)
classes = ['adonis','american snoot','an 88','banded peacock','beckers white','black hairstreak',
          'cabbage white','chestnut','clodius parnassian','clouded sulphur',
          'copper tail','crecent','crimson patch','eastern coma','gold banded',
          'great eggfly','grey hairstreak','indra swallow','julia','large marble',
          'malachite','mangrove skipper','metalmark','monarch','morning cloak',
          'orange oakleaf','orange tip','orchard swallow','painted lady','paper kite',
          'peacock','pine white','pipevine swallow','purple hairstreak','question mark',
          'red admiral','red spotted purple','scarce swallow','silver spot skipper',
          'sixspot burnet','skipper','sootywing','southern dogface','straited queen',
          'two barred flasher','ulyses','viceroy','wood satyr','yellow swallow tail','zebra long wing']

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Butterfly Recognition")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("Model_2")
    return model
    
with st.spinner('Loading model into memmory...'):
    model = load_model()


def load_and_prep_image(image):
    """
    Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape,, color_channels)
    """
    # Read in the image
    # img = tf.io.read_file(filename)
    # Decode the read file into a tensor
    image = tf.image.decode_image(image)
    # Resize the image  
    image = tf.image.resize(image, size=IMAGE_SHAPE)
    #Grayscale
    if image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
        # Rescale the image (getting all values between 0 & 1)
        # image = image/255

    return image

def url_uploader():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.text("Provide Url for Butterfly Recognition")
    
    path = st.text_input("Enter image Url to classify...", "https://github.com/Hrushi11/Butterfly_Recognition/blob/main/images/monarch_test.jpg?raw=true")
    if path is not None:
        content = requests.get(path).content

        st.write("Predicted Butterfly :")
        with st.spinner("Classifying....."):
            img = load_and_prep_image(content)
            label = model.predict(tf.expand_dims(img, axis=0))
            st.write(classes[int(tf.argmax(tf.squeeze(label).numpy()))])

        st.write("")
        image = Image.open(BytesIO(content))
        st.image(image, caption="Classifying the Butterfly", use_column_width=True)


def file_Uploader():
    file = st.file_uploader("Upload file", type=["png", "jpeg", "jpg"])
    show_file = st.empty()

    if not file:
        show_file.info("Upload a picture of the Butterfly you want to predict.")
        return

    content = file.getvalue()

    st.write("Predicted Butterfly :")
    with st.spinner("Classifying....."):
         img = load_and_prep_image(content)
         label = model.predict(tf.expand_dims(img, axis=0))
         st.write(classes[int(tf.argmax(tf.squeeze(label).numpy()))])
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption="Classifying the Butterfly", use_column_width=True)

st.sidebar.header('Choose how you want to upload a file')
# st.sidebar.write("URL - To predict from a link, \n File Upload - To predict from a file present on your device")
function = st.sidebar.selectbox('URL or File Upload',('URL','File Upload'))

if function == 'URL':
    url_uploader()
else :
    file_Uploader()
