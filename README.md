

![14.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/LNIOy8V_w0loMrVMtTyIK.png)

# **Fashion-Product-Usage**

> **Fashion-Product-Usage** is a vision-language model fine-tuned from **google/siglip2-base-patch16-224** using the **SiglipForImageClassification** architecture. It classifies fashion product images based on their intended usage context.

```py
Classification Report:
              precision    recall  f1-score   support

      Casual     0.8529    0.9716    0.9084     34392
      Ethnic     0.8365    0.7528    0.7925      3208
      Formal     0.7246    0.3006    0.4250      2345
        Home     0.0000    0.0000    0.0000         1
       Party     0.0000    0.0000    0.0000        29
Smart Casual     0.0000    0.0000    0.0000        67
      Sports     0.7157    0.1848    0.2938      4004
      Travel     0.0000    0.0000    0.0000        26

    accuracy                         0.8458     44072
   macro avg     0.3912    0.2762    0.3024     44072
weighted avg     0.8300    0.8458    0.8159     44072
```

The model predicts one of the following usage categories:

- **0:** Casual  
- **1:** Ethnic  
- **2:** Formal  
- **3:** Home  
- **4:** Party  
- **5:** Smart Casual  
- **6:** Sports  
- **7:** Travel

---

# **Run with Transformers 🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```

---

# **Intended Use**

This model can be used for:

- **Product tagging in e-commerce catalogs**  
- **Context-aware product recommendations**  
- **Fashion search optimization**  
- **Data annotation for training recommendation engines**
