import os
import glob
import subprocess
import logging
import sys

"""
process_all_webinars.py

Purpose:
This script automates the processing of multiple webinar report pairs using webinar-report-merge.py.
It iterates through CSV files in the ./input directory, matches attendee and registration files,
and processes each pair using webinar-report-merge.py.

Key features:
- Scans ./input directory for CSV files
- Matches attendee and registration file pairs
- Runs webinar-report-merge.py for each valid pair
- Implements debug logging and error handling
- Provides verbose console output

Usage:
Place this script in the same directory as webinar-report-merge.py and run it to process all webinar reports in ./input.
Ensure that webinar-report-merge.py is in the same directory and that ./input contains the CSV files to be processed.
"""

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_matching_files(input_dir):
    """
    Find matching attendee and registration CSV files in the input directory.

    Args:
    input_dir (str): Path to the input directory containing CSV files.

    Returns:
    list: A list of tuples, each containing paths to matching attendee and registration files.
    """
    logger.debug(f"Scanning directory: {input_dir}")
    
    # Get all attendee CSV files
    attendee_files = glob.glob(os.path.join(input_dir, "attendee_*.csv"))
    matching_pairs = []

    for attendee_file in attendee_files:
        # Extract the identifier from the attendee filename
        identifier = attendee_file.split("attendee_")[1]
        
        # Construct the expected registration filename
        registration_file = os.path.join(input_dir, f"registration_{identifier}")
        
        # Check if the matching registration file exists
        if os.path.exists(registration_file):
            matching_pairs.append((attendee_file, registration_file))
            logger.debug(f"Matched pair found: {attendee_file} and {registration_file}")
        else:
            logger.warning(f"No matching registration file found for {attendee_file}")

    return matching_pairs

def process_file_pair(attendee_file, registration_file):
    """
    Process a pair of attendee and registration files using webinar-report-merge.py.

    Args:
    attendee_file (str): Path to the attendee CSV file.
    registration_file (str): Path to the registration CSV file.

    Returns:
    bool: True if processing was successful, False otherwise.
    """
    logger.info(f"Processing files: {attendee_file} and {registration_file}")
    
    try:
        # Construct the command to run webinar-report-merge.py
        command = [sys.executable, "webinar-report-merge.py", registration_file, attendee_file]
        
        # Run the command and capture output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Log the output from webinar-report-merge.py
        logger.debug(f"webinar-report-merge.py output:\n{result.stdout}")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing files: {e}")
        logger.error(f"webinar-report-merge.py error output:\n{e.stderr}")
        return False

def main():
    """
    Main function to process all matching webinar report pairs in the input directory.
    """
    input_dir = "./input"
    
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist.")
        return

    # Find matching file pairs
    matching_pairs = find_matching_files(input_dir)
    
    if not matching_pairs:
        logger.warning("No matching file pairs found.")
        return

    # Process each matching pair
    successful_pairs = 0
    for attendee_file, registration_file in matching_pairs:
        if process_file_pair(attendee_file, registration_file):
            successful_pairs += 1

    # Log summary
    logger.info(f"Processing complete. {successful_pairs} out of {len(matching_pairs)} pairs processed successfully.")

if __name__ == "__main__":
    main()
