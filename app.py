import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Fashion-Product-Usage"  # Replace with your actual model path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    0: "Casual",
    1: "Ethnic",
    2: "Formal",
    3: "Home",
    4: "Party",
    5: "Smart Casual",
    6: "Sports",
    7: "Travel"
}

def classify_usage(image):
    """Predicts the usage type of a fashion product."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {id2label[i]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Gradio interface
iface = gr.Interface(
    fn=classify_usage,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Usage Prediction Scores"),
    title="Fashion-Product-Usage",
    description="Upload a fashion product image to predict its intended usage (Casual, Formal, Party, etc.)."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
