# tracker/views.py

from django.shortcuts import render, redirect
from django.db.models import Count, Sum
from django.utils import timezone
from .models import Employee, AbsenceReason, AbsenceLog
import joblib
import pandas as pd
import json

# --- 1. MODEL LOADING ---
# Load the ThresholdOptimizer you exported from Colab
try:
    MODEL_PATH = 'tracker/ml_model/to_bmi_model.joblib'
    TO_BMI_MODEL = joblib.load(MODEL_PATH)
    print("--- Fairlearn 'to_bmi' model loaded successfully. ---")
except FileNotFoundError:
    print("--- WARNING: 'to_bmi_model.joblib' not found. Predictions will fail. ---")
    TO_BMI_MODEL = None
except Exception as e:
    print(f"--- ERROR loading model: {e} ---")
    TO_BMI_MODEL = None

# --- Tab 1: Dashboard View (No Change) ---
def dashboard_view(request):
    # This part is the same
    current_absences = AbsenceLog.objects.filter(status='ABSENT').order_by('-date_logged')
    
    # --- THIS IS THE CHANGE ---
    # We now Sum 'predicted_hours' and rename the annotated field to 'total_hours'
    reason_data = AbsenceLog.objects.values('reason__description') \
                                      .annotate(total_hours=Sum('predicted_hours')) \
                                      .order_by('-total_hours')
    
    # Filter out reasons with 0 or negative hours, as they shouldn't be on the chart
    reason_data = reason_data.filter(total_hours__gt=0)
    
    # Update variable names to be clearer
    chart_labels = [item['reason__description'] for item in reason_data]
    chart_values = [item['total_hours'] for item in reason_data]
    # --- END OF CHANGE ---

    context = {
        'current_absences': current_absences,
        'chart_labels': json.dumps(chart_labels),
        'chart_data': json.dumps(chart_values), # Pass the new chart_values
        'page': 'dashboard' 
    }
    return render(request, 'tracker/1_dashboard.html', context)

# --- Tab 2: Log Absence View (THE FINAL FIX) ---
def log_absence_view(request):
    context = {
        'employees': Employee.objects.all(),
        'reasons': AbsenceReason.objects.all(),
        'prediction_result': None,
        'page': 'log_absence'
    }

    if request.method == 'POST' and TO_BMI_MODEL:
        try:
            # 1. Get data from form
            employee_id = int(request.POST.get('employee_id'))
            reason_code = int(request.POST.get('reason_code'))
            
            # 2. Get the full employee object from DB
            employee = Employee.objects.get(employee_id=employee_id)
            reason = AbsenceReason.objects.get(reason_code=reason_code)
            
            # 3. Build the feature dictionary (X)
            # This contains ONLY the features your inner model was trained on.
            # NO 'Education', NO 'bmi_cat'.
            feature_dict = {
                'Reason for absence': reason_code,
                'Month of absence': timezone.now().month,
                'Day of the week': timezone.now().isoweekday() + 1, # Mon=2, Tue=3...
                'Seasons': (timezone.now().month%12 + 3)//3, # Simple season logic
                'Transportation expense': employee.transportation_expense,
                'Distance from Residence to Work': employee.distance_from_residence_to_work,
                'Service time': employee.service_time,
                'Age': employee.age,
                'Work load Average/day ': employee.work_load_average_day, 
                'Hit target': employee.hit_target,
                'Body mass index': employee.body_mass_index,
            }

            # 4. Create the main DataFrame (X)
            X_pred = pd.DataFrame([feature_dict])

            # 5. Create the SEPARATE sensitive_features Series
            # We get the employee's BMI and categorize it, just as in Colab.
            bins = [0, 18.5, 25, 30, 100]
            labels = ['underweight', 'normal weight', 'over weight', 'obese']
            
            # Create a pandas Series, not a DataFrame column
            sensitive_features_series = pd.cut(
                [employee.body_mass_index], # Pass BMI as a list
                bins=bins, 
                labels=labels, 
                right=False
            )
            
            # tracker/views.py

# ... (inside log_absence_view) ...

            # 6. Make prediction
            predicted_hours = TO_BMI_MODEL.predict(
                X_pred, 
                sensitive_features=sensitive_features_series
            )[0]
            
            predicted_hours = max(0, float(predicted_hours)) # Keep full precision

            if predicted_hours > 0:
                # 7. Save to AbsenceLog (ONLY if > 0)
                AbsenceLog.objects.create(
                    employee=employee,
                    reason=reason,
                    predicted_hours=predicted_hours,
                    status='ABSENT'
                )
                
                # Set the success context
                context['prediction_result'] = {
                    'name': employee.full_name,
                    'hours': predicted_hours
                }
            else:
                # Set a NEW context variable for 0-hour predictions
                context['prediction_result_zero'] = {
                    'name': employee.full_name,
                }

        except Exception as e:
            context['error'] = f"Prediction failed: {e}. Check terminal for details."
            print(f"--- PREDICTION ERROR: {e} ---")

    return render(request, 'tracker/2_log_absence.html', context)

# --- Tab 3: Salaries View (No Change) ---
def salaries_view(request):
    STANDARD_WORK_HOURS = 160.0
    employees = Employee.objects.all()
    salary_data = []
    
    current_month = timezone.now().month
    current_year = timezone.now().year

    for emp in employees:
        absent_hours_agg = AbsenceLog.objects.filter(
            employee=emp,
            date_logged__month=current_month,
            date_logged__year=current_year
        ).aggregate(total_hours=Sum('predicted_hours'))
        
        total_absent_hours = absent_hours_agg['total_hours'] or 0.0
        expected_work_hours = max(0, STANDARD_WORK_HOURS - float(total_absent_hours))
        expected_compensation = expected_work_hours * float(emp.hourly_rate)
        
        salary_data.append({
            'name': emp.full_name,
            'total_absent_hours': total_absent_hours, # Pass the full float
            'expected_work_hours': round(expected_work_hours, 1),
            'expected_compensation': round(expected_compensation, 2)
        })

    context = {
        'salary_data': salary_data,
        'page': 'salaries'
    }
    return render(request, 'tracker/3_salaries.html', context)

# --- Tab 4: About the Model View (No Change) ---
def about_model_view(request):
    context = {'page': 'about'}
    return render(request, 'tracker/4_about.html', context)