# import_data.py

import os
import django
import pandas as pd
import sys # Import sys to exit the script on error

# 1. SETUP: LOAD DJANGO ENVIRONMENT
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'absenteeism_project.settings')
django.setup()

# 2. IMPORT YOUR MODELS
from tracker.models import AbsenceReason, Employee

# 3. DEFINE YOUR DATA AND MAPPINGS
try:
    # --- IMPORTANT: Make sure this path is correct ---
    DATASET_PATH = '~/Absenteeism_at_work.csv' 
    df = pd.read_csv(DATASET_PATH, sep=';')
    print(f"--- Successfully loaded DataFrame from {DATASET_PATH} ---")

    # --- NEW DEBUG STEP 1: Print all columns from your CSV ---
    print("\n--- DataFrame Columns (from your CSV file) ---")
    print(df.columns.to_list())
    print("-------------------------------------------------\n")

except FileNotFoundError:
    print(f"ERROR: Dataset file not found at {DATASET_PATH}")
    print("Please update the DATASET_PATH variable in this script.")
    sys.exit()
except Exception as e:
    print(f"Error loading DataFrame: {e}")
    sys.exit()

# Map of all reasons (This should be correct)
REASON_MAP = {
    0: 'No Reason Given', 1: 'Certain infectious and parasitic diseases',
    2: 'Neoplasms', 3: 'Diseases of the blood',
    4: 'Endocrine, nutritional and metabolic diseases', 5: 'Mental and behavioural disorders',
    6: 'Diseases of the nervous system', 7: 'Diseases of the eye and adnexa',
    8: 'Diseases of the ear and mastoid process', 9: 'Diseases of the circulatory system',
    10: 'Diseases of the respiratory system', 11: 'Diseases of the digestive system',
    12: 'Diseases of the skin and subcutaneous tissue', 13: 'Diseases of the musculoskeletal system',
    14: 'Diseases of the genitourinary system', 15: 'Pregnancy, childbirth and the puerperium',
    16: 'Certain conditions originating in the perinatal period', 17: 'Congenital malformations, deformations',
    18: 'Symptoms, signs and abnormal clinical findings', 19: 'Injury, poisoning',
    20: 'External causes of morbidity and mortality', 21: 'Factors influencing health status',
    22: 'Patient follow-up', 23: 'Medical consultation', 24: 'Blood donation',
    25: 'Laboratory examination', 26: 'Unjustified absence', 27: 'Physiotherapy',
    28: 'Dental consultation',
}

# --- POTENTIAL ERROR LOCATION ---
# Compare the printout above with these keys. They must match perfectly.
# I am keeping the trailing space on 'Work load...' as it's the most common version.
COLUMN_MAP = {
    'ID': 'employee_id',
    'Age': 'age',
    'Education': 'education',
    'Body mass index': 'body_mass_index',
    'Transportation expense': 'transportation_expense',
    'Distance from Residence to Work': 'distance_from_residence_to_work',
    'Service time': 'service_time',
    'Work load Average/day ': 'work_load_average_day', # <-- Check this key carefully!
    'Hit target': 'hit_target',
}

# --- NEW DEBUG STEP 2: Validate column map against the CSV ---
csv_columns = df.columns.to_list()
map_keys = list(COLUMN_MAP.keys())

missing_keys = [key for key in map_keys if key not in csv_columns]

if missing_keys:
    print(f"--- ERROR: SCRIPT STOPPED ---")
    print("Your CSV is missing columns that `COLUMN_MAP` needs.")
    print("Missing column(s):")
    for key in missing_keys:
        print(f"  - '{key}'")
    
    print("\nACTION: Look at the 'DataFrame Columns' printout above.")
    print("Then, edit the `COLUMN_MAP` dictionary in this script to match your CSV.")
    sys.exit() # Stop the script
else:
    print("--- Column map validated. All keys found in CSV. ---")


# 4. DEFINE THE IMPORT FUNCTION
def populate_database():
    
    print("\n--- Starting Data Import ---")

    # --- Step A: Populate AbsenceReason Table ---
    print("Populating AbsenceReason table...")
    reasons_created_count = 0
    for code, desc in REASON_MAP.items():
        reason, created = AbsenceReason.objects.get_or_create(
            reason_code=code,
            defaults={'description': desc}
        )
        if created:
            reasons_created_count += 1
    print(f"Created {reasons_created_count} new absence reasons.")

    # --- Step B: Populate Employee Table ---
    print("Processing unique employees from DataFrame...")
    df_unique_employees = df.drop_duplicates(subset='ID', keep='last')
    print(f"Found {len(df_unique_employees)} unique employees.")
    
    employees_created_count = 0
    employees_updated_count = 0

    for index, row in df_unique_employees.iterrows():
        try:
            employee_data = {}
            for df_col, model_field in COLUMN_MAP.items():
                employee_data[model_field] = row[df_col]

            employee_data['full_name'] = f"Employee {row['ID']}"
            employee_data['hourly_rate'] = 30.00 

            employee, created = Employee.objects.update_or_create(
                employee_id=row['ID'],  
                defaults=employee_data     
            )

            if created:
                employees_created_count += 1
            else:
                employees_updated_count += 1

        except Exception as e:
            # This will catch any other unexpected errors
            print(f"  ERROR: Failed to import Employee {row['ID']}. Error: {e}")

    print("\n--- Import Complete ---")
    print(f"Created {employees_created_count} new employees.")
    print(f"Updated {employees_updated_count} existing employees.")

# 5. RUN THE FUNCTION
if __name__ == '__main__':
    populate_database()