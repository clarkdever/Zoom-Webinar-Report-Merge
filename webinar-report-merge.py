import pandas as pd
import re
import os
import argparse
import logging
import csv
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------
# ------------------------------------------------------
# Script Overview:
# This script merges two reports (registration and attendee reports) into a single CSV output. 
# The reports, 'regrep.csv' and 'attrep.csv', contain metadata of variable length at the top, 
# followed by a consistent header structure. The script dynamically identifies the header row
# using known column names and processes the data accordingly.
#
# Objectives:
# 1. Identify the header rows in both files dynamically based on fixed column names 
#    (e.g., 'First Name', 'Last Name', 'Email', etc.). The header row can be located at different
#    positions in each file (e.g., Row 6 for regrep, Row 15 for attrep).
# 2. Clean the email addresses by removing spaces and converting them to lowercase, 
#    then use these emails as identifiers for merging the two reports.
# 3. Handle cases where the two files have different numbers of columns by padding the data 
#    appropriately before merging.
# 4. Extract metadata like event name, scheduled time, and event duration from specific rows 
#    (e.g., Row 4 in regrep and Row 4 in attrep).
# 5. Convert certain fields, such as the 'Attended' field from YES/NO to True/False.
# 6. Generate an output CSV file with the following columns:
#    - First Name, Last Name, Email, Organization, Registration Time, Source Name (from regrep)
#    - Event Name, Scheduled Time, Event Duration (from regrep metadata)
#    - Attended, Time in Session, Country/Region (from attrep)
# 7. The output file should be named based on the event (e.g., {{event name}}.csv).
#    If no output file name is specified, the script will default to {{event name}}.csv.
#
# Requirements:
# Ensure that any fields with commas, such as dates, are properly escaped to prevent 
# misalignment in the CSV output. This can be handled by ensuring that date fields are
# wrapped in quotes before writing to CSV.
#
# How to Run:
# 1. Ensure you have Anaconda or Miniconda installed.
# 2. Create and activate the Conda environment using the provided environment.yml file:
#    conda env create -f environment.yml
#    conda activate webinar-report-merge
# 3. Once the environment is activated, execute the script with the following command:
#    python webinar-report-merge.py regrep.csv attrep.csv
# 
# Note: Make sure you're in the correct directory containing the script and CSV files.
# ------------------------------------------------------
# ------------------------------------------------------

# Function to clean email addresses by trimming whitespace and lowercasing them
def clean_email(email):
    if pd.isnull(email):
        return None
    return email.strip().lower()

# Function to convert the 'Attended' field from "YES" or "NO" to boolean True or False
def convert_attended(attended):
    return True if attended.strip().upper() == "YES" else False

# Function to manually extract event details (like event name and scheduled time) from the CSV by reading specific rows
def extract_event_details_manual(file_path, event_row):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract event name and scheduled time based on the file's structure
    event_name = lines[3].split(',')[0].strip()  # Example: 'Foundations of Visual Thinking'
    scheduled_time = lines[3].split(',')[2].strip()  # Example: 'Sep 25, 2024 12:30 PM'
    
    return event_name, scheduled_time

# Function to find the header row in a file based on expected headers
def find_header_row(file_path, expected_headers):
    logging.debug(f"Searching for header row in {file_path}")
    logging.debug(f"Expected headers: {expected_headers}")
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            logging.debug(f"Line {i}: {line.strip()}")
            if all(header in line for header in expected_headers):
                logging.info(f"Header row found at line {i} in {file_path}")
                return i
    logging.warning(f"Header row not found in {file_path}")
    return None  # Return None instead of raising an exception

# Function to extract event details from a file based on the header row
def extract_event_details(file_path, header_row):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    event_name = lines[header_row - 2].split(',')[0].strip()
    scheduled_time = lines[header_row - 2].split(',')[2].strip()
    return event_name, scheduled_time

def convert_to_athena_date_format(date_string):
    """
    Convert various date string formats to Athena DB's preferred format (YYYY-MM-DD HH:MM:SS).
    This function handles multiple input formats and ensures consistency in the output.

    Args:
    date_string (str): The input date string to be converted.

    Returns:
    str: The date in YYYY-MM-DD HH:MM:SS format if successfully converted, otherwise an empty string.
    """
    if pd.isna(date_string) or date_string == '--':
        logger.debug(f"Empty or invalid date string: '{date_string}'")
        return ''
    
    formats = [
        '%m/%d/%Y %I:%M:%S %p',
        '%b %d, %Y %I:%M %p',
        '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %I:%M %p',
        '%Y-%m-%d %I:%M:%S %p',
        '%b %d, %Y %H:%M:%S',  # New format for joinTime and leaveTime
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_string, fmt)
            result = dt.strftime('%Y-%m-%d %H:%M:%S')
            logger.debug(f"Successfully converted '{date_string}' to '{result}' using format '{fmt}'")
            return result
        except ValueError:
            continue
    
    logger.warning(f"Failed to convert date string: '{date_string}'. Tried formats: {formats}")
    return ''

def convert_is_guest(value):
    """
    Convert 'Is Guest' values to boolean TRUE or FALSE.
    
    Args:
    value (str): The input value from the 'Is Guest' column.

    Returns:
    bool: TRUE if the value is 'Yes', FALSE otherwise.
    """
    if pd.isna(value) or value in ['--', '']:
        return False
    return value.strip().lower() == 'yes'

def convert_time_in_session(value):
    """
    Convert timeInSession to minutes if it's a valid number, otherwise return 0.
    
    Args:
    value: The input value from the 'timeInSession' column.

    Returns:
    int: The time in session in minutes, or 0 if invalid.
    """
    if pd.isna(value) or value == '' or value == '--':
        return 0
    try:
        # If the value is already in minutes, just return it as an integer
        return int(float(value))
    except ValueError:
        # If it's not a simple number, it might be in HH:MM:SS format
        try:
            parts = value.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 60 + minutes + (1 if seconds > 0 else 0)
            elif len(parts) == 2:
                minutes, seconds = map(int, parts)
                return minutes + (1 if seconds > 0 else 0)
        except:
            pass
    
    logger.warning(f"Unable to convert timeInSession value to minutes: {value}")
    return 0

# Main function that processes both CSV files and merges them, adding fixed patches for formatting and parsing
def process_csv_fixed(regrep_file, attrep_file):
    logging.info(f"Processing files: regrep={regrep_file}, attrep={attrep_file}")

    # Expected headers for each file (for finding the header row)
    regrep_headers = ['First Name', 'Last Name', 'Email']
    attrep_headers = ['First Name', 'Last Name', 'Email', 'Attended']

    # Find header rows
    regrep_header_row = find_header_row(regrep_file, regrep_headers)
    attrep_header_row = find_header_row(attrep_file, attrep_headers)

    logging.info(f"regrep_file: {regrep_file}, header row: {regrep_header_row}")
    logging.info(f"attrep_file: {attrep_file}, header row: {attrep_header_row}")

    if regrep_header_row is None:
        raise ValueError(f"Header row not found in {regrep_file}")
    if attrep_header_row is None:
        raise ValueError(f"Header row not found in {attrep_file}")

    # Read CSV files using csv.reader to handle quoted fields correctly
    with open(regrep_file, 'r') as f:
        reader = csv.reader(f)
        regrep_data = [row for row in reader]
    
    with open(attrep_file, 'r') as f:
        reader = csv.reader(f)
        attrep_data = [row for row in reader]

    # Debug: Print the header row and the first data row
    logger.debug(f"Header row: {regrep_data[regrep_header_row]}")
    logger.debug(f"First data row: {regrep_data[regrep_header_row+1]}")
    
    # Count the number of columns in the header and data
    header_columns = len(regrep_data[regrep_header_row])
    data_columns = len(regrep_data[regrep_header_row+1])
    
    logger.debug(f"Number of columns in header: {header_columns}")
    logger.debug(f"Number of columns in data: {data_columns}")

    # If there's a mismatch, adjust the header
    if header_columns != data_columns:
        logger.warning(f"Column mismatch detected. Adjusting header.")
        if header_columns < data_columns:
            # Add generic column names if header has fewer columns
            regrep_data[regrep_header_row] += [f"Column_{i}" for i in range(header_columns, data_columns)]
        else:
            # Remove extra column names if header has more columns
            regrep_data[regrep_header_row] = regrep_data[regrep_header_row][:data_columns]

    # Create DataFrame with adjusted header
    regrep = pd.DataFrame(regrep_data[regrep_header_row+1:], columns=regrep_data[regrep_header_row])

    # Debug: Print the first few rows of regrep
    logger.debug(f"First few rows of regrep:\n{regrep.head()}")

    # Now let's handle the attrep file
    logger.debug(f"Header row for attrep: {attrep_data[attrep_header_row]}")
    logger.debug(f"First data row for attrep: {attrep_data[attrep_header_row+1]}")
    
    # Count the number of columns in the header and data for attrep
    attrep_header_columns = len(attrep_data[attrep_header_row])
    attrep_data_columns = len(attrep_data[attrep_header_row+1])
    
    logger.debug(f"Number of columns in attrep header: {attrep_header_columns}")
    logger.debug(f"Number of columns in attrep data: {attrep_data_columns}")

    # If there's a mismatch, adjust the header for attrep
    if attrep_header_columns != attrep_data_columns:
        logger.warning(f"Column mismatch detected in attrep. Adjusting header.")
        if attrep_header_columns < attrep_data_columns:
            attrep_data[attrep_header_row] += [f"Column_{i}" for i in range(attrep_header_columns, attrep_data_columns)]
        else:
            attrep_data[attrep_header_row] = attrep_data[attrep_header_row][:attrep_data_columns]

    # Create DataFrame for attrep with adjusted header
    attrep = pd.DataFrame(attrep_data[attrep_header_row+1:], columns=attrep_data[attrep_header_row])

    # Debug: Print the first few rows of attrep
    logger.debug(f"First few rows of attrep:\n{attrep.head()}")

    # Clean email addresses
    regrep['Email'] = regrep['Email'].apply(clean_email)
    attrep['Email'] = attrep['Email'].apply(clean_email)

    # Convert 'Attended' column
    attrep['Attended'] = attrep['Attended'].apply(convert_attended)

    # Extract information from regrep_data
    event_name = regrep_data[3][0]  # Event name from the first column of the fourth row
    scheduled_time = regrep_data[3][2]  # Scheduled time from the third column of the fourth row
    duration = regrep_data[3][3]  # Duration from the fourth column of the fourth row

    logger.debug(f"Event Name: {event_name}")
    logger.debug(f"Scheduled Time: {scheduled_time}")
    logger.debug(f"Duration: {duration}")

    # Merge the DataFrames
    merged_report = pd.merge(regrep, attrep, on='Email', how='outer', suffixes=('', '_y'))

    # Function to combine rows and calculate time in session
    def combine_rows(group):
        combined = group.iloc[0]
        for col in group.columns:
            if col.endswith('_y'):
                base_col = col[:-2]
                combined[base_col] = group[base_col].fillna(group[col]).iloc[0]
            elif pd.isna(combined[col]):
                combined[col] = group[col].dropna().iloc[0] if not group[col].dropna().empty else combined[col]
        
        # Calculate Time in Session
        if 'Join Time' in combined and 'Leave Time' in combined:
            if combined['Join Time'] != '--' and combined['Leave Time'] != '--':
                try:
                    join_time = pd.to_datetime(combined['Join Time'])
                    leave_time = pd.to_datetime(combined['Leave Time'])
                    time_in_session = (leave_time - join_time).total_seconds() / 60
                    combined['Time in Session'] = f"{time_in_session:.0f}"
                except:
                    combined['Time in Session'] = '0'
            else:
                combined['Time in Session'] = '0'
        else:
            combined['Time in Session'] = '0'
        
        return combined

    # Group by Email and apply the combine_rows function
    merged_report = merged_report.groupby('Email', as_index=False).apply(combine_rows)

    # Remove duplicate columns
    columns_to_remove = [col for col in merged_report.columns if col.endswith('_y')]
    merged_report = merged_report.drop(columns=columns_to_remove)

    # Add the extracted information to the merged report
    merged_report['Event Name'] = event_name
    merged_report['Scheduled Time'] = scheduled_time
    merged_report['Duration'] = duration

    # After merging and processing the data, but before writing to CSV:

    # 1. Remove the "Approval Status" column
    merged_report = merged_report.drop(columns=['Approval Status'], errors='ignore')

    # 2. Replace Null values in "Attended" column with FALSE
    merged_report['Attended'] = merged_report['Attended'].fillna(False)

    # 3. Replace Null values in "Source Name" column with "Unknown"
    merged_report['Source Name'] = merged_report['Source Name'].fillna("Unknown")
    merged_report['Source Name'] = merged_report['Source Name'].replace('', "Unknown")

    # 4. Replace 'nan' with blank in country/regionName
    merged_report['Country/Region Name'] = merged_report['Country/Region Name'].replace('nan', '')

    # 5. Rename 'User Name (Original Name)' to 'User Name'
    merged_report = merged_report.rename(columns={'User Name (Original Name)': 'User Name'})

    # 6. Replace 'nan' with blank in User Name
    merged_report['User Name'] = merged_report['User Name'].replace('nan', '')

    # Update the required columns list
    required_columns = [
        'Email', 'First Name', 'Last Name', 'Organization', 'Registration Time', 'Source Name',
        'Attended', 'User Name', 'Join Time', 'Leave Time', 'Time in Session', 'Is Guest',
        'Country/Region Name', 'Event Name', 'Scheduled Time', 'Duration'
    ]

    # Ensure all required columns are present and filled
    for col in required_columns:
        if col not in merged_report.columns:
            merged_report[col] = ''
        elif col in ['Country/Region Name', 'User Name'] and merged_report[col].isna().any():
            merged_report[col] = merged_report[col].fillna('')

    # Reorder columns to match the desired output
    merged_report = merged_report[required_columns]

    # Remove spaces and convert column headers to camel case
    def to_camel_case(column_name):
        words = column_name.split()
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

    merged_report.columns = [to_camel_case(col) for col in merged_report.columns]

    # Debug: Show merged report information
    logger.debug(f"Merged report columns: {merged_report.columns.tolist()}")
    logger.debug(f"First few rows of merged report:\n{merged_report.head()}")

    # Convert date columns to Athena DB format for easier ingestion into Superset
    date_columns = ['registrationTime', 'scheduledTime', 'joinTime', 'leaveTime']
    for col in date_columns:
        if col in merged_report.columns:
            logger.info(f"Converting dates in column: {col}")
            merged_report[col] = merged_report[col].apply(convert_to_athena_date_format)
            logger.info(f"Finished converting dates in column: {col}")

    # Convert 'isGuest' column
    if 'isGuest' in merged_report.columns:
        merged_report['isGuest'] = merged_report['isGuest'].apply(convert_is_guest)

    # Convert timeInSession column
    if 'timeInSession' in merged_report.columns:
        logger.info("Converting timeInSession column")
        merged_report['timeInSession'] = merged_report['timeInSession'].apply(convert_time_in_session)
        merged_report['timeInSession'] = merged_report['timeInSession'].astype(int)  # Ensure integer type
        logger.info("Finished converting timeInSession column")

    # After processing all columns, log a sample of the data
    logger.debug("Sample of processed data:")
    logger.debug(merged_report[date_columns + ['timeInSession']].head().to_string())

    # Ensure all columns are string type before writing to CSV
    for column in merged_report.columns:
        merged_report[column] = merged_report[column].astype(str)

    # Generate output file name
    output_file = f"{event_name}.csv"

    # Create the output directory if it doesn't exist
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    output_file = os.path.join(output_dir, output_file)

    # Save the merged report
    merged_report.to_csv(output_file, index=False)
    logger.info(f"Merged report saved to {output_file}")

    return output_file

# Main function to handle command-line arguments, allowing for flexible input/output options
if __name__ == "__main__":
    # Argument parser for handling input files and output file arguments from the command line
    parser = argparse.ArgumentParser(description='Combine regrep and attrep CSV files into a final report.')
    parser.add_argument('regrep_file', help='Path to the regrep CSV file.')
    parser.add_argument('attrep_file', help='Path to the attrep CSV file.')
    args = parser.parse_args()

    # Debug information
    logging.info(f"regrep_file: {args.regrep_file}")
    logging.info(f"attrep_file: {args.attrep_file}")
    output_file = process_csv_fixed(args.regrep_file, args.attrep_file)
    print(f"Merged report saved as: {output_file}")
