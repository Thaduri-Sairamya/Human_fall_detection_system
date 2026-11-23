# Human Fall Detection — Streamlit starter project

## Included files
- `app.py` — Streamlit application for image and video inference.
- `requirements.txt` — Python dependencies.
- `best.pt` — **Not included**; add your own trained YOLO weights.
- `outputs/` — Folder reserved for outputs.

## Setup (Windows / VS Code)

1. Extract this ZIP.
2. Copy your trained model weights into this folder and name the file `best.pt`.
3. Open a terminal in this folder.
4. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   streamlit run app.py
   ```

The app normally opens at http://localhost:8501.

## Important model requirement

This application assumes a custom-trained Ultralytics YOLO model whose class labels include a fall-related name, such as `fall`, `falling`, or `fallen`. Labels such as `not fall`, `no fall`, `non fall`, `normal`, and `standing` are treated as non-fall labels.

If your model uses different class names, edit `is_fall_class()` in `app.py`.

A generic person-detection model cannot by itself determine whether a person has fallen. If your model detects only `person`, you need a fall-classification model or an appropriate pose/temporal fall-detection pipeline.

## Notes
- The video metric counts frames containing a fall-class detection, not distinct fall incidents.
- The app processes uploaded videos locally and may take time for long or high-resolution videos.
- This starter project does not send real emergency notifications.
- Keep `best.pt` private if it contains proprietary training work; do not upload large model weights to a public repository without checking the hosting limits and license.
