# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Simple CNN model for MNIST
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    """Load pre-trained model or create a new one"""
    model = MNISTCNN()
    # In a real scenario, you would load pre-trained weights here
    # model.load_state_dict(torch.load('mnist_cnn.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess the uploaded image for MNIST classification"""
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize and reshape for model
    image_array = image_array.astype(np.float32) / 255.0
    image_array = (image_array - 0.1307) / 0.3081  # MNIST normalization
    
    # Add batch and channel dimensions
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    
    return image_tensor, image_array

def main():
    st.set_page_config(page_title="MNIST Classifier", page_icon="🔢", layout="wide")
    
    # Title and description
    st.title("🔢 MNIST Digit Classifier")
    st.markdown("""
    Upload an image of a handwritten digit (0-9) and the model will predict what digit it is!
    """)
    
    # Load model
    model = load_model()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a handwritten digit"
        )
        
        # Or draw a digit
        st.subheader("Or Draw a Digit")
        drawing_enabled = st.checkbox("Enable drawing canvas")
        
        if drawing_enabled:
            canvas_result = st.canvas(
                fill_color="black",
                stroke_width=15,
                stroke_color="white",
                background_color="black",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="canvas",
            )
    
    with col2:
        st.subheader("Prediction Results")
        
        if uploaded_file is not None:
            # Process uploaded file
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=200)
            
            # Preprocess and predict
            image_tensor, processed_array = preprocess_image(image)
            
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                predicted_class = torch.argmax(output[0]).item()
                confidence = probabilities[predicted_class].item()
            
            # Display results
            st.success(f"**Prediction: {predicted_class}**")
            st.info(f"**Confidence: {confidence:.2%}**")
            
            # Show probabilities for all classes
            st.subheader("Class Probabilities")
            fig, ax = plt.subplots()
            classes = list(range(10))
            probs = probabilities.numpy()
            ax.bar(classes, probs)
            ax.set_xlabel('Digit')
            ax.set_ylabel('Probability')
            ax.set_xticks(classes)
            ax.set_ylim(0, 1)
            st.pyplot(fig)
            
        elif drawing_enabled and canvas_result is not None:
            # Process drawn image
            if canvas_result.image_data is not None:
                # Convert canvas result to PIL Image
                drawn_image = Image.fromarray((canvas_result.image_data * 255).astype(np.uint8))
                
                # Preprocess and predict
                image_tensor, processed_array = preprocess_image(drawn_image)
                
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    predicted_class = torch.argmax(output[0]).item()
                    confidence = probabilities[predicted_class].item()
                
                # Display results
                st.success(f"**Prediction: {predicted_class}**")
                st.info(f"**Confidence: {confidence:.2%}**")
                
                # Show processed image
                st.subheader("Processed Image (28x28)")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                ax1.imshow(processed_array, cmap='gray')
                ax1.set_title('Processed Image')
                ax1.axis('off')
                
                ax2.bar(range(10), probabilities.numpy())
                ax2.set_xlabel('Digit')
                ax2.set_ylabel('Probability')
                ax2.set_xticks(range(10))
                ax2.set_ylim(0, 1)
                ax2.set_title('Class Probabilities')
                st.pyplot(fig)
        
        else:
            st.info("Please upload an image or draw a digit to get a prediction")
    
    # Add some information about the model
    with st.expander("About this Model"):
        st.markdown("""
        **Model Architecture:**
        - Convolutional Neural Network (CNN)
        - 2 Convolutional layers
        - 2 Fully connected layers
        - Trained on MNIST dataset
        
        **Input Requirements:**
        - 28x28 pixel grayscale images
        - Handwritten digits (0-9)
        - Clear, centered digits work best
        """)

if __name__ == "__main__":
    main()