# 🔬 SpectraQual 2.0 — PCB Defect Detection Demo

> AI-powered visual inspection for printed circuit board images using MobileNetV2 transfer learning, built with Streamlit.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00C853?style=flat-square)

---

## 📸 What It Does

SpectraQual 2.0 is a **prototype AI inspection tool** that classifies PCB images as:

- ✅ **Good** — board passes visual inspection
- ❌ **Defective** — board flagged with a likely defect type

Designed for use with **phone camera photos of printed PCB papers** (no real hardware required), making it ideal for rapid prototyping and demo environments.

**Detected defect categories (by confidence threshold):**

| Confidence | Flagged Defect Types |
|---|---|
| ≥ 80% | Short Circuit, Missing Component |
| 65–79% | Solder Bridge, Cold Joint, Lifted Pad |
| 50–64% | Possible Contamination, Marginal Solder |

---

## 🖥️ Demo UI

```
┌─────────────────────────────────────────┐
│        SpectraQual 2.0 PCB Demo         │
│  PCB Visual Inspection · MobileNetV2   │
├─────────────────────────────────────────┤
│  [ Upload Image ]  or  [ 📷 Webcam ]    │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │       Processed PCB Image        │   │
│  └──────────────────────────────────┘   │
│                                         │
│  [ 🔬 Analyse PCB ]                     │
│                                         │
│  ❌ Defective · 85.3% confidence        │
│  ⚠️ Short Circuit  Missing Component    │
│  ████████████████░░░░  Confidence Bar   │
└─────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-username/spectraqual-pcb-demo.git
cd spectraqual-pcb-demo
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run pcb_demo.py
```

The app opens automatically at `http://localhost:8501`.

---

## 📦 Requirements

```
streamlit>=1.32.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
```

> **Note:** Use `opencv-python-headless` (not `opencv-python`) for server/headless environments. Use `opencv-python` if you need GUI windows on your local machine.

Save the above as `requirements.txt` or install individually:

```bash
pip install streamlit torch torchvision pillow opencv-python-headless numpy
```

---

## 🗂️ Project Structure

```
spectraqual-pcb-demo/
│
├── pcb_demo.py          # Main Streamlit app (single file, fully self-contained)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ How It Works

```
Phone Camera Photo
        │
        ▼
┌───────────────────┐
│  OpenCV Filters   │  ← Optional grayscale + edge enhancement
│  (PCB shine fix)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Resize 224×224  │
│ ImageNet Normalize│  ← Standard torchvision transforms
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  MobileNetV2      │  ← ImageNet pretrained backbone (frozen)
│  Backbone         │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Binary Head      │  ← Linear(1280, 2) → [Good, Defective]
│  Dropout(0.2)     │
└────────┬──────────┘
         │
         ▼
   Softmax Probabilities
   → Label + Confidence %
   → Defect type tags
```

### Model Architecture

| Component | Detail |
|---|---|
| Backbone | MobileNetV2 (ImageNet weights, frozen) |
| Classifier head | `Dropout(0.2)` → `Linear(1280 → 2)` |
| Input size | 224 × 224 RGB |
| Normalisation | ImageNet mean/std |
| Output | 2-class softmax (Good / Defective) |
| Threshold | 0.50 |

---

## 🔧 Sidebar Options

| Option | Default | Effect |
|---|---|---|
| Grayscale | Off | Converts to greyscale to reduce copper/solder shine |
| Edge Enhance | On | Blends Laplacian edges for sharper joint boundaries |

---

## 📷 Input Modes

**File Upload**
- Accepts JPG / JPEG / PNG
- Drag-and-drop or click to browse

**Webcam (`st.camera_input`)**
- Toggle the checkbox to switch to live camera
- Click the camera button to capture a frame
- Works in browser (requires camera permission)

---

## ⚠️ Demo Mode Notice

> The binary classification head ships **without fine-tuned weights** — it is randomly initialised. A small stochastic perturbation is applied at inference time to produce plausible demo outputs.
>
> **For production use:** replace the head initialisation block with your own fine-tuned checkpoint:

```python
# Load your trained weights
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features, 2),
)
model.load_state_dict(torch.load("your_pcb_weights.pth", map_location="cpu"))
model.eval()
```

---

## 🏋️ Training Your Own Model (Next Steps)

To move from demo → production:

1. **Collect data** — photograph printed good/defective PCB papers under consistent lighting.
2. **Label** — organise into `data/good/` and `data/defective/` folders.
3. **Fine-tune** — unfreeze the last few MobileNetV2 layers and train the head:

```python
# Unfreeze last convolutional block for fine-tuning
for name, param in model.named_parameters():
    if "features.18" in name or "classifier" in name:
        param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)
criterion = nn.CrossEntropyLoss()
```

4. **Save weights** → load in `pcb_demo.py` as shown above.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a pull request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Howard et al., Google Brain
- [torchvision](https://github.com/pytorch/vision) — PyTorch pretrained models
- [Streamlit](https://streamlit.io) — rapid ML app framework
- [OpenCV](https://opencv.org) — image preprocessing

---

<p align="center">Built with ❤️ for rapid PCB prototyping · SpectraQual 2.0</p>
