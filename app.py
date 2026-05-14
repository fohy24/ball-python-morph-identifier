import os
import json
import torch
from torch import nn
from torchvision import models
from torchvision.transforms import v2
from huggingface_hub import hf_hub_download
import gradio as gr

labels = [
    'Pastel',
    'Clown',
    'Yellow Belly',
    'Enchi',
    'Piebald',
    'Leopard',
    'Orange Dream',
    'Fire',
    'Mojave',
    'Spotnose',
    'Banana',
    'Desert Ghost',
    'Black Pastel',
    'Hypo',
    'Normal',
    'Pinstripe',
    'GHI',
    'Lesser',
    'Cinnamon',
    'Red Stripe',
    'Black Head',
    'Super Pastel',
    'Chocolate',
    'Axanthic (VPI)',
    'Cypress',
    'Vanilla',
    'Gravel',
    'Butter',
    'Ultramel',
    'Asphalt',
    'Calico',
    'Stranger',
    'Spider',
    'Lavender Albino',
    'Hurricane',
    'Mahogany',
    'Albino'
    ]

num_labels = len(labels)

# Toggle between optimized per-class thresholds and a fixed 0.5 cutoff
USE_OPTIMIZED_THRESHOLD = True

# If using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hf_token = os.getenv('HF_token')
model_path = hf_hub_download(repo_id="samfhy/morphmarket_model", filename="model_v1_epoch20.pt", token=hf_token)
checkpoint = torch.load(model_path, map_location=device)

# Load per-class optimized thresholds
thresholds_path = hf_hub_download(repo_id="samfhy/morphmarket_model", filename="optimal_thresholds_model2_v1.json", token=hf_token)
with open(thresholds_path, 'r') as f:
    optimal_thresholds = json.load(f)

new_layers = nn.Sequential(
    nn.LazyLinear(2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.LazyLinear(num_labels)
    )

IMAGE_SIZE = checkpoint['image_size']
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.6007, 0.5679, 0.5206], std=[0.2411, 0.2392, 0.2479]),
    ])

tta_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.RandomHorizontalFlip(p=1.0),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.6007, 0.5679, 0.5206], std=[0.2411, 0.2392, 0.2479]),
    ])

efficientnet = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')
efficientnet.classifier = new_layers
efficientnet.load_state_dict(checkpoint['model_state_dict'])
efficientnet.to(device).eval()

def predict(img):
    if img is None:
        return {}
    img = img.convert('RGB')

    base_input = transform(img).unsqueeze(0).to(device)
    flipped_input = tta_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        base_probs = torch.sigmoid(efficientnet(base_input))
        flipped_probs = torch.sigmoid(efficientnet(flipped_input))

    predicted_probs = ((base_probs + flipped_probs) / 2.0).cpu().flatten().tolist()

    prediction_dict = {}
    for i, label in enumerate(labels):
        thresh = optimal_thresholds.get(label, 0.5) if USE_OPTIMIZED_THRESHOLD else 0.5
        if predicted_probs[i] >= thresh:
            prediction_dict[label] = predicted_probs[i]

    return prediction_dict


with gr.Blocks(title='Ball Python Morph Identifier') as demo:
    gr.Markdown("# Ball Python Morph Identifier")
    gr.Markdown("Upload or paste an image of your ball python to identify its morphs!")
    gr.Markdown("""
        If you're unfamiliar with snakes, ball pythons come in various patterns and colors,
        called *morphs*, which can be difficult to distinguish without expert knowledge.
        This tool automatically identifies these unique variations, making identification accessible to everyone.
        Try selecting one of the examples and click "Identify Morphs" to see how it works!
        """)

    with gr.Accordion("Click here to show all the morphs that can be predicted", open=False):
        gr.Markdown("""
        Albino, Asphalt, Banana, Black Head, Black Pastel, Butter, Calico, Chocolate, Cinnamon, Clown,
        Desert Ghost, Enchi, Fire, GHI, Gravel, Hypo, Leopard, Lesser, Mojave, Normal,
        Orange Dream, Pastel, Piebald, Pinstripe, Red Stripe, Spider, Spotnose, Super Pastel, Vanilla, Yellow Belly
        """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload/Paste Image")
            gr.Examples(
                examples=[
                    ["enchi_albino_clown.png", "Enchi, Albino, Clown"],
                    ["mojave_ghi.png", "Mojave, GHI"],
                    ["hypo_banana_pastel_enchi.png", "Hypo, Banana, Pastel, Enchi"],
                    ["yb_pastel_gravel.png", "Yellow Belly, Pastel, Gravel"],
                    ["ivory.png", "Super Yellow Belly"]
                    ],
                inputs=[img_input]
            )
            predict_btn = gr.Button("Identify Morphs", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(label="Predicted Morphs")

    predict_btn.click(fn=predict, inputs=[img_input], outputs=label_output)

demo.launch()
