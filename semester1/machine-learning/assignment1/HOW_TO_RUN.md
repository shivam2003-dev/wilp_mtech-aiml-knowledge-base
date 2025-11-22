# How to Run ML Assignment 1 Notebook

## Prerequisites

1. **Activate Virtual Environment:**
```bash
cd /Users/shivamkumar/Desktop/shivam/ml_assignment
source venv/bin/activate
```

2. **Install Required Packages:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter ipykernel
```

Or install from requirements:
```bash
pip install -r ml_requirements.txt
```

## Running the Notebook

### Option 1: Using Jupyter Notebook (Recommended)

```bash
# Make sure you're in the correct directory
cd /Users/shivamkumar/Desktop/shivam/ml_assignment

# Activate virtual environment
source venv/bin/activate

# Start Jupyter Notebook
jupyter notebook
```

Then:
1. Open `ML_Assignment_1_Solution.ipynb` in the browser
2. Click `Cell → Run All` to run all cells
3. Or run cells one by one using `Shift + Enter`

### Option 2: Using Jupyter Lab

```bash
cd /Users/shivamkumar/Desktop/shivam/ml_assignment
source venv/bin/activate
jupyter lab
```

### Option 3: Using VS Code

1. Open VS Code in the directory: `/Users/shivamkumar/Desktop/shivam/ml_assignment`
2. Install Jupyter extension if not already installed
3. Open `ML_Assignment_1_Solution.ipynb`
4. Select the kernel: `Python 3.x.x ('venv': venv)`
5. Run cells using the play button or `Shift + Enter`

## Important: Working Directory

**⚠️ CRITICAL:** The notebook expects to be run from:
```
/Users/shivamkumar/Desktop/shivam/ml_assignment
```

This is because the datasets are loaded from:
- `datasets/bike_train.csv`
- `datasets/bike_test.csv`
- `datasets/sampleSubmission.csv`

## Verify Setup

Before running, verify datasets exist:
```bash
ls -lh datasets/
```

Should show:
- bike_train.csv (~657KB)
- bike_test.csv (~132KB)
- sampleSubmission.csv (~49KB)

## Troubleshooting

### Issue: "No such file or directory: 'datasets/bike_train.csv'"

**Solution:** Make sure you're running the notebook from the correct directory:
```bash
cd /Users/shivamkumar/Desktop/shivam/ml_assignment
pwd  # Should show: /Users/shivamkumar/Desktop/shivam/ml_assignment
```

### Issue: Import errors

**Solution:** Install missing packages:
```bash
pip install <package_name>
```

### Issue: Kernel not found in Jupyter

**Solution:** Install ipykernel and register the venv:
```bash
source venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

Then select this kernel in Jupyter.

## Expected Output

After running all cells, you should see:
- ✓ All 12 questions answered
- Visualizations for Q2 and Q8
- Model comparison table in Q7
- `submission.csv` file generated

## Quick Test

Test if everything works:
```bash
cd /Users/shivamkumar/Desktop/shivam/ml_assignment
source venv/bin/activate
python3 -c "import pandas as pd; df = pd.read_csv('datasets/bike_train.csv'); print(f'✓ Dataset loaded: {df.shape}')"
```

