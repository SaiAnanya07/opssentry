# OpsSentry - Quick Setup Guide

## ✅ Installation Complete!

All dependencies have been installed successfully.

## 🚀 Running OpsSentry

### Option 1: Use Batch Files (Easiest for Windows)

#### Run Complete Pipeline:
```cmd
run_pipeline.bat
```

This will:
1. Fetch GitHub Actions data
2. Create labeled datasets
3. Preprocess data
4. Train models

#### Start Web Server:
```cmd
start_server.bat
```

Then open `http://localhost:5000` in your browser.

### Option 2: Manual Commands (PowerShell)

Set the Python path first, then run commands:

```powershell
$env:PYTHONPATH = "c:\Users\yedit\OneDrive\Desktop\final_yr_project\opssentry"

# Step 1: Collect data
python scripts\fetch_runs.py --max-pages 5

# Step 2: Create datasets
python scripts\data_collector.py --github-only --max-pages 5

# Step 3: Preprocess
python scripts\preprocess.py --source github

# Step 4: Train models
python scripts\train_model.py

# Step 5: Start server
python app.py
```

### Option 3: Use Python Script

```powershell
python quickstart.py
```

## 📝 Important Notes

### About the "No runs found" Warning

If you see "No runs found", it means your GitHub repository doesn't have many workflow runs yet. This is normal for new repositories.

**Solutions:**
1. **Use sample data**: The system can work with minimal data for testing
2. **Add more workflows**: Create some GitHub Actions workflows in your repo
3. **Use LogHub data**: Download system logs as alternative training data

### Using LogHub Data

If you don't have enough GitHub Actions data:

```powershell
$env:PYTHONPATH = "c:\Users\yedit\OneDrive\Desktop\final_yr_project\opssentry"
python scripts\download_loghub.py --dataset HDFS_v1
python scripts\data_collector.py --loghub-only
python scripts\preprocess.py --source loghub
python scripts\train_model.py
```

## 🔧 Troubleshooting

### Module Import Errors

If you get `ModuleNotFoundError: No module named 'config'`:

**Solution**: Always set PYTHONPATH before running scripts:

```powershell
$env:PYTHONPATH = "c:\Users\yedit\OneDrive\Desktop\final_yr_project\opssentry"
```

Or use the batch files which do this automatically.

### Virtual Environment

Your venv is outside the project, which is fine. Just make sure dependencies are installed:

```powershell
pip install -r requirements.txt
```

### Not Enough Data

If you get errors about insufficient data:

1. **Increase data collection**:
   ```powershell
   python scripts\fetch_runs.py --max-pages 20
   ```

2. **Use LogHub datasets** (recommended for testing):
   ```powershell
   python scripts\download_loghub.py --all --limit 3
   ```

## 📊 Testing the System

### Test with Sample Prediction

Once the server is running, test the API:

```powershell
# In a new PowerShell window
Invoke-RestMethod -Uri "http://localhost:5000/api/health" -Method Get
```

Or use the web dashboard at `http://localhost:5000`

### Manual Testing

```powershell
$env:PYTHONPATH = "c:\Users\yedit\OneDrive\Desktop\final_yr_project\opssentry"
python
```

Then in Python:
```python
from models.predictor import Predictor

# This will work after training
predictor = Predictor()
result = predictor.predict_pipeline_run({
    "name": "Test Pipeline",
    "build_duration": 450.0,
    "num_jobs": 3
})
print(result)
```

## 🎯 Recommended Workflow

For your final year project demonstration:

1. **Collect sample data**:
   ```cmd
   run_pipeline.bat
   ```

2. **Start the server**:
   ```cmd
   start_server.bat
   ```

3. **Demo the dashboard**:
   - Open `http://localhost:5000`
   - Fill in the prediction form
   - Show the results and recommendations

4. **Show the API**:
   - Use Postman or curl to demonstrate API endpoints
   - Show real-time predictions

## 📁 Project Structure

```
opssentry/
├── run_pipeline.bat      ← Run this first!
├── start_server.bat      ← Then run this!
├── quickstart.py         ← Alternative Python script
├── app.py                ← Web server
├── config.py             ← Configuration
├── requirements.txt      ← Dependencies (already installed)
├── README.md             ← Full documentation
├── scripts/              ← All processing scripts
├── models/               ← Trained models (after training)
├── data/                 ← Collected data
├── templates/            ← Web UI
└── static/               ← CSS/JS
```

## 🆘 Need Help?

Check the main README.md for detailed documentation:
```powershell
notepad README.md
```

Or check the walkthrough document in the brain folder for complete implementation details.

---

**You're all set! Start with `run_pipeline.bat` 🚀**
