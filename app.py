import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

# Configuration and model definition
CONFIG = dict(
    seed = 42,
    model_name = 'tf_efficientnet_b4_ns',
    train_batch_size = 16,
    valid_batch_size = 32,
    img_size = 256,
    epochs = 5,
    learning_rate = 1e-4,
    scheduler = 'CosineAnnealingLR',
    min_lr = 1e-6,
    T_max = 100,
    T_0 = 25,
    warmup_epochs = 0,
    weight_decay = 1e-6,
    n_accumulate = 1,
    n_fold = 5,
    num_classes = 1,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    competition = 'PetFinder',
    _wandb_kernel = 'deb'
)

class PawpularityModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super(PawpularityModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.fc = nn.LazyLinear(CONFIG['num_classes'])
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, images, meta):
        features = self.model(images)                 # Extract features
        features = self.dropout(features)
        features = torch.cat([features, meta], dim=1) # Concatenate metadata
        output = self.fc(features)                    # Predict Pawpularity
        return output



# Load the model
model = PawpularityModel(CONFIG['model_name'])
model.load_state_dict(torch.load('model_new.pth', map_location=CONFIG['device']))
model.to(CONFIG['device'])
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


st.title("Pawpularity Score Prediction 🐾")
st.write("Project by Shreya Sivakumar-20BCE1794")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image and prepare dummy metadata (replace with actual metadata handling)
    image = transform(image).unsqueeze(0).to(CONFIG['device'])
    meta = torch.zeros((1, 12)).to(CONFIG['device'])  


    with torch.no_grad():
        output = model(image, meta)
        pawpularity_score = output.item()

    st.markdown(f"<h2 style='text-align: center; color: black;'>🐾 Pawpularity Score: {pawpularity_score}</h1>", unsafe_allow_html=True)
    st.markdown("""
    ---
    Copyright © 2024 Shreya Sivakumar. All rights reserved.
    """)
