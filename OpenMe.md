# Deployment

A Streamlit app that loads a trained CNN-ViT Hybrid checkpoint and classifies
an uploaded photo of a waste item into one of five categories: `plastic`,
`metal`, `glass`, `paper_cardboard`, or `trash`.

## Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Get a trained checkpoint. Either:
   - Train your own using the code in `New_ModelTraining.ipynb`, then copy the
     resulting `cnn_vit_hybrid_5class_merged.pth` into `model_weights/`, or
   - Use an existing checkpoint file, placed at
     `model_weights/cnn_vit_hybrid_5class_merged.pth`.

   The folder structure should look like:

   ```
   deployment/
   ├── app.py
   ├── model.py
   ├── requirements.txt
   └── model_weights/
       └── cnn_vit_hybrid_5class_merged.pth
   ```

   If your checkpoint has a different name or location, edit
   `CHECKPOINT_PATH` near the top of `app.py`.

## Running locally

```
streamlit run app.py
```

This opens the app in your browser (default: `http://localhost:8501`).
Upload a photo and the app displays the predicted class, confidence, and
the probability for every class.

## Deploying to Streamlit Community Cloud

1. Push this repository (or just the `deployment/` folder, restructured as
   its own repo root) to GitHub. Note that model checkpoints are typically
   too large for a normal GitHub push if they exceed 100MB — use
   [Git LFS](https://git-lfs.com/) for the `.pth` file if needed, or host it
   externally (e.g. Hugging Face Hub, cloud storage) and download it at
   startup instead of committing it directly.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and
   create a new app pointing at your repository and `app.py`.
3. Streamlit Cloud installs `requirements.txt` automatically and runs the
   app — no server setup needed.

## How it works

- `model.py` defines the `Enhanced-CNNViT` architecture (ResNet50 + DeiT-Small,
  fused through a shared classifier head) and a `load_model()` helper that
  reconstructs the model from a checkpoint and loads its trained weights.
- `app.py` is the Streamlit UI: it loads the model once (cached across
  requests), preprocesses an uploaded image the same way the training
  script's validation transform did (resize to 224x224, normalize with
  ImageNet statistics), and displays the resulting class probabilities.

## Troubleshooting

-"No model checkpoint found" — the app couldn't find a `.pth` file at
  `CHECKPOINT_PATH`. Confirm the file exists at that exact path, or update
  the path in `app.py`.
- Slow predictions on CPU — this model runs two backbones (a CNN and a
  transformer) per image, so CPU inference is noticeably slower than GPU.
  A GPU is not required to run the app, only to speed it up.
- Import error for `timm` — make sure `pip install -r requirements.txt`
  completed successfully; `timm` is required to construct both backbones.
