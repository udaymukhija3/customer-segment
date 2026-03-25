# Customer Segmentation Studio

This frontend is a static demo surface for the customer segmentation API. It is designed to feel like a portfolio-ready control room rather than a bare form wrapper.

## Highlights

- Live API health and model telemetry
- Persona presets for fast demos
- Rich prediction stories with confidence, drivers, and actions
- Segment atlas powered by the `/segments` endpoint
- Responsive single-file implementation with no build step

## Run locally

Start the API first:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then serve the frontend:

```bash
cd frontend
python3 -m http.server 3000
```

Open [http://localhost:3000](http://localhost:3000).

## Notes

- The API base URL is editable from the UI and persisted in local storage.
- The demo pulls from `/health`, `/model/info`, `/segments`, and `/predict`.
- Because the UI is static, it is easy to host on GitHub Pages, Netlify, or an S3 bucket.
