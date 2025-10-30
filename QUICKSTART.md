# Quick Start Guide for Flask App

## Step 1: Generate Model Files

Before running the Flask app, you need to generate the required model files by running the notebook:

1. Open `analysis_model.ipynb` in VS Code or Jupyter
2. Run all cells from top to bottom
3. Make sure the last cell that saves the best model runs successfully

This will create the following files:
- `gender_label_encoder.pkl`
- `diabetic_label_encoder.pkl`
- `smoker_label_encoder.pkl`
- `region_label_encoder.pkl`
- `scaler.pkl`
- `best_model.pkl`

## Step 2: Install Flask Dependencies

Install Flask and other required packages:

```bash
pip install flask
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

## Step 3: Verify Setup

Run the check script to verify all files are present:

```bash
python check_setup.py
```

## Step 4: Run the Flask App

Start the Flask development server:

```bash
python app.py
```

You should see output like:
```
 * Running on http://127.0.0.1:5000
```

## Step 5: Access the Web App

Open your web browser and go to:
```
http://localhost:5000
```

## Troubleshooting

### Import Error: No module named 'flask'
- Solution: Run `pip install flask`

### FileNotFoundError: [Errno 2] No such file or directory: 'best_model.pkl'
- Solution: Run all cells in `analysis_model.ipynb` to generate model files

### Model Loading Error
- Solution: Make sure you ran the corrected cell that saves `top_model` (not `top_idx`)

## Testing the API

You can test the API endpoint using curl or Python:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "gender": "male", "bmi": 25.5, "bloodpressure": 120, "diabetic": "No", "children": 2, "smoker": "No", "region": "northeast"}'
```

## Next Steps

- Customize the styling in `templates/index.html` and `templates/result.html`
- Add more validation
- Deploy to a cloud platform (Heroku, AWS, Azure, etc.)
- Add authentication if needed
