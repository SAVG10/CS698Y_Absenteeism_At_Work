# import_data.py

import os
import django
import pandas as pd

# 1. SETUP: LOAD DJANGO ENVIRONMENT
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'absenteeism_project.settings')
django.setup()

# 2. IMPORT YOUR MODELS
from tracker.models import AbsenceReason, Employee

# 3. DEFINE YOUR DATA AND MAPPINGS
# --- IMPORTANT: ADJUST THIS ---
try:
    DATASET_PATH = '~/Absenteeism_at_work.csv' # Point this to your file
    df = pd.read_csv(DATASET_PATH, sep=';')
    print(f"Successfully loaded DataFrame from {DATASET_PATH}")
except FileNotFoundError:
    print(f"ERROR: Dataset file not found at {DATASET_PATH}")
    print("Please update the DATASET_PATH variable in this script.")
    exit()
except Exception as e:
    print(f"Error loading DataFrame: {e}")
    exit()

# Map of all reasons (This is correct)
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

# --- THIS IS THE MAIN CHANGE ---
# The map now *only* includes static employee features.
COLUMN_MAP = {
    'ID': 'employee_id',
    'Age': 'age',
    'Education': 'education',
    'Body mass index': 'body_mass_index',
    'Transportation expense': 'transportation_expense',
    'Distance from Residence to Work': 'distance_from_residence_to_work',
    'Service time': 'service_time',
    'Work load Average/day ': 'work_load_average_day', # Note the trailing space
    'Hit target': 'hit_target',
}

# 4. DEFINE THE IMPORT FUNCTION
def populate_database():
    
    print("--- Starting Data Import ---")

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
            # Prepare a dictionary of data for the Employee model
            employee_data = {}
            for df_col, model_field in COLUMN_MAP.items():
                # All columns in the new map are direct copies
                employee_data[model_field] = row[df_col]

            # Add the placeholder fields
            employee_data['full_name'] = f"Employee {row['ID']}"
            employee_data['hourly_rate'] = 40  # Default value

            # Use update_or_create to add new employees or update existing ones
            employee, created = Employee.objects.update_or_create(
                employee_id=row['ID'],  # Key
                defaults=employee_data     # Data to update/create
            )

            if created:
                employees_created_count += 1
            else:
                employees_updated_count += 1

        except Exception as e:
            print(f"  ERROR: Failed to import Employee {row['ID']}. Error: {e}")

    print("\n--- Import Complete ---")
    print(f"Created {employees_created_count} new employees.")
    print(f"Updated {employees_updated_count} existing employees.")

# 5. RUN THE FUNCTION
if __name__ == '__main__':
    populate_database()