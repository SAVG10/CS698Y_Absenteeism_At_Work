# tracker/views.py

from django.shortcuts import render, redirect
from django.db.models import Count, Sum
from django.utils import timezone
from .models import Employee, AbsenceReason, AbsenceLog
import joblib
import pandas as pd
import json

# --- 1. MODEL LOADING (CHANGED) ---
# We now load ONE file: your 'ThresholdOptimizer' model.
# It already contains the preprocessor.
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
    current_absences = AbsenceLog.objects.filter(status='ABSENT').order_by('-date_logged')
    reason_counts = AbsenceLog.objects.values('reason__description') \
                                      .annotate(count=Count('reason')) \
                                      .order_by('-count')
    chart_labels = [item['reason__description'] for item in reason_counts]
    chart_data = [item['count'] for item in reason_counts]

    context = {
        'current_absences': current_absences,
        'chart_labels': json.dumps(chart_labels),
        'chart_data': json.dumps(chart_data),
        'page': 'dashboard'
    }
    return render(request, 'tracker/1_dashboard.html', context)

# --- Tab 2: Log Absence View (CHANGED) ---
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
            
            # 3. Build the feature dictionary for the model
            # This MUST match the columns your ColumnTransformer was trained on
            feature_dict = {
                'Reason for absence': reason_code,
                'Month of absence': timezone.now().month,
                'Day of the week': timezone.now().isoweekday() + 1, # Mon=2, Tue=3...
                'Seasons': (timezone.now().month%12 + 3)//3, # Simple season logic
                'Transportation expense': employee.transportation_expense,
                'Distance from Residence to Work': employee.distance_from_residence_to_work,
                'Service time': employee.service_time,
                'Age': employee.age,
                # CRITICAL: This key MUST have the trailing space to match your Colab notebook
                'Work load Average/day ': employee.work_load_average_day, 
                'Hit target': employee.hit_target,
                'Education': employee.education,
                'Body mass index': employee.body_mass_index,
            }

            # 4. Create DataFrame
            X_pred = pd.DataFrame([feature_dict])

            # 5. (NEW) Re-create the sensitive feature 'bmi_cat'
            # These are the same bins/labels from your Colab
            bins = [0, 18.5, 25, 30, 100]
            labels = ['underweight', 'normal weight', 'over weight', 'obese']
            # We create the 'bmi_cat' column for our 1-row DataFrame
            X_pred['bmi_cat'] = pd.cut(
                X_pred['Body mass index'], 
                bins=bins, 
                labels=labels, 
                right=False
            )
            
            # 6. (NEW) Extract the sensitive feature for the .predict() method
            sensitive_features = X_pred['bmi_cat']

            # 7. (CHANGED) Make prediction
            # We call .predict() on the ThresholdOptimizer
            # It takes X AND the sensitive_features
            # We no longer need the PREPROCESSOR.transform() line
            predicted_hours = TO_BMI_MODEL.predict(X_pred, sensitive_features=sensitive_features)[0]
            
            # Ensure prediction is non-negative
            predicted_hours = max(0, round(float(predicted_hours), 1))

            # 8. Save to AbsenceLog
            AbsenceLog.objects.create(
                employee=employee,
                reason=reason,
                predicted_hours=predicted_hours,
                status='ABSENT'
            )
            
            context['prediction_result'] = {
                'name': employee.full_name,
                'hours': predicted_hours
            }

        except Exception as e:
            context['error'] = f"Prediction failed: {e}. Ensure all employee data is correct."
            print(f"ERROR: {e}")

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
            'total_absent_hours': round(total_absent_hours, 1),
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