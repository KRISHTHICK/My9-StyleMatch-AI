import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize models
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample fashion database
fashion_db = {
    "outfit_1": "White oversized shirt with blue ripped jeans and white sneakers.",
    "outfit_2": "Black crop top with high-waisted trousers and chunky boots.",
    "outfit_3": "Beige trench coat over a black turtleneck and plaid skirt.",
    "outfit_4": "Yellow sundress with floral print and wedge heels.",
}

st.set_page_config(page_title="StyleMatch AI", layout="wide")
st.title("👗 StyleMatch AI - Find Similar Fashion Styles")

uploaded_image = st.file_uploader("📸 Upload an Outfit Image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Generating fashion description..."):
        inputs = caption_processor(images=img, return_tensors="pt")
        output = caption_model.generate(**inputs)
        description = caption_processor.decode(output[0], skip_special_tokens=True)

    st.subheader("📝 Detected Fashion Description")
    st.write(description)

    # Generate embedding
    query_embedding = embedding_model.encode(description, convert_to_tensor=True)

    # Search for similar outfit
    results = {}
    for name, desc in fashion_db.items():
        db_embedding = embedding_model.encode(desc, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(query_embedding, db_embedding).item()
        results[name] = (desc, round(similarity, 2))

    # Sort and display results
    st.subheader("🔍 Similar Outfits")
    top_matches = sorted(results.items(), key=lambda x: x[1][1], reverse=True)[:3]
    for name, (desc, score) in top_matches:
        st.markdown(f"**{name}** - Match Score: `{score}`")
        st.write(desc)
        st.markdown("---")
