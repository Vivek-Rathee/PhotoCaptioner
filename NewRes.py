import os
import json
import boto3
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import io

# Set page configuration for dark theme and layout
st.set_page_config(page_title="Photo Captioner", layout="wide", initial_sidebar_state="expanded")

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

os.environ["AWS_PROFILE"] = "AlphaController"

# Rekognition and Bedrock clients with additional config
from botocore.config import Config

config = Config(
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    },
    connect_timeout=10,
    read_timeout=30
)

rekognition_client = boto3.client("rekognition", region_name="us-east-1", config=config)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1", config=config)

modelID = "amazon.titan-text-express-v1"

def generate_caption(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_bytes = img_byte_arr.getvalue()

    try:
        response = rekognition_client.detect_labels(
            Image={'Bytes': img_bytes},
            MaxLabels=10,
            MinConfidence=80
        )
        labels = [label['Name'] for label in response['Labels']]
        caption = "a photo of " + ", ".join(labels)
        return caption
    except Exception as e:
        st.error(f"Error detecting labels: {e}")
        return None

def refine_caption(caption, platform):
    if platform == "LinkedIn":
        input_text = f"Make the following caption more formal for LinkedIn: {caption}"
    elif platform == "Instagram":
        input_text = f"Make the following caption more casual for Instagram: {caption}"

    body = json.dumps({
        "inputText": input_text,
        "textGenerationConfig": {
            "maxTokenCount": 50,
            "stopSequences": [],
            "temperature": 0.1,
            "topP": 1
        }
    }).encode('utf-8')

    try:
        response = bedrock_client.invoke_model(
            modelId=modelID,
            contentType="application/json",
            accept="application/json",
            body=body
        )

        response_text = response['body'].read().decode('utf-8')
        response_body = json.loads(response_text)

        refined_caption = response_body['results'][0]['outputText'].strip()
        refined_caption = refined_caption.replace("Here is the revised caption:", "").strip()
        return refined_caption
    except Exception as e:
        st.error(f"Error refining caption: {e}")
        return None

st.title("AI Powered - Photo Captioner")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True, width=500)

    platform = st.selectbox("Platform to post", ["Instagram", "LinkedIn"])
    
    if st.button("Generate Caption"):
        st.write("Generating Caption...")
        caption = generate_caption(image)

        if caption:
            refined_caption = refine_caption(caption, platform)
            if refined_caption:
                st.success(refined_caption)
            else:
                st.error("Failed to refine caption.")
        else:
            st.error("Failed to generate caption.")
