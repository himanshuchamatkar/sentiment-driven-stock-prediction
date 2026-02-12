import subprocess
import time
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Ticker is currently hardcoded for the test, but can be configured in .env later
TICKER = "RELIANCE.NS" 

def run_pipeline(script_path):
    """Runs a Python script as a separate process."""
    try:
        # Use subprocess to execute the script as a module
        # check=True raises an error if the script fails (non-zero exit code)
        result = subprocess.run(['python', '-m', script_path], 
                                capture_output=True, text=True, check=True)
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] --- {script_path.upper()} SUCCESS ---")
        # Print the stdout from the script for visibility
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] !!! {script_path.upper()} FAILED !!!")
        print(f"Error Code: {e.returncode}")
        print(f"Stderr: {e.stderr}")
        # Re-raise the exception to stop the overall pipeline if a critical failure occurs
        raise

def daily_job():
    """
    The main daily pipeline:
    1. Fetch latest OHLC data.
    2. Engineer features and update database.
    3. Run ensemble prediction and save results.
    """
    print(f"\n{'='*50}\nSTARTING DAILY AI PREDICTION PIPELINE for {TICKER}\n{'='*50}")

    # 1. Fetch & Save Latest Data
    run_pipeline('utils.data_fetcher') 
    time.sleep(1) # Small pause to ensure DB access is released

    # 2. Calculate Features & Label
    run_pipeline('utils.feature_engineer')
    time.sleep(1)

    # 3. Generate Prediction and Save Results
    run_pipeline('scripts.run_prediction')

    print(f"\n{'='*50}\nDAILY PIPELINE COMPLETE\n{'='*50}")


def train_job():
    """Retrains the models periodically (e.g., weekly/monthly)."""
    print(f"\n{'*'*50}\nSTARTING WEEKLY MODEL RETRAINING\n{'*'*50}")
    run_pipeline('scripts.train_xgboost')
    run_pipeline('scripts.train_lstm')
    print(f"\n{'*'*50}\nMODEL RETRAINING COMPLETE\n{'*'*50}")

if __name__ == '__main__':
    # Run the full training pipeline once to simulate weekly retraining
    train_job() 

    # Start the scheduler (but we comment out start() and run daily_job immediately for testing)
    scheduler = BlockingScheduler()

    # scheduler.add_job(daily_job, 'cron', day_of_week='mon-fri', hour=8, minute=30)
    
    print("\nStarting scheduler simulation... (Running daily job once for final test)")
    daily_job() # Run once immediately for final verification
    
    # scheduler.start()