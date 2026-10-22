# WattWise AI Dashboard

A Streamlit dashboard using the trained models and dataset exported from the WattWise AI Colab notebook.

## Included
- `app.py` — dashboard application
- `requirements.txt` — pinned runtime dependencies
- `models/` — trained models, scalers, metadata, and chart data

## Run locally
Use Python 3.11 if possible, then run:

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Upload this project to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

Keep the `models/` folder in the repository. The model files must remain alongside the app in that folder.

## Important notes
- Models were saved with scikit-learn 1.6.1; the requirement pins that version because joblib model files can be version-sensitive.
- The dashboard presents historical data from 2017 and model estimates. It is not connected to a live utility grid.
- Cluster IDs are arbitrary labels and should be interpreted using the displayed profiles.
