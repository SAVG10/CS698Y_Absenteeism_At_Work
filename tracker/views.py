# tracker/views.py

from django.shortcuts import render, redirect
from django.db.models import Count, Sum
from django.utils import timezone
from .models import Employee, AbsenceReason, AbsenceLog
import joblib
import pandas as pd
import json

# Recommendations aligned to REASON_MAP codes (import_data.py)
REASON_RECOMMENDATIONS = {
    0: "Investigate unreported absences and reinforce timely reporting with manager follow-ups.",
    1: "Promote hygiene, sick-leave usage, and vaccination to reduce infectious spread.",
    2: "Support flexible schedules for treatment and reinforce medical leave accommodations.",
    3: "Coordinate with healthcare needs and provide schedule flexibility for treatments.",
    4: "Offer wellness coaching for diabetes/thyroid management and allow flexible breaks.",
    5: "Strengthen EAP access, mental-health days, and workload balance to reduce burnout.",
    6: "Provide quiet spaces and flexible hours for migraines/neurological conditions.",
    7: "Improve screen ergonomics, enforce screen-breaks, and expand vision benefits.",
    8: "Ensure timely ENT access and improve noise control/PPE where relevant.",
    9: "Offer heart-health screenings and encourage activity through wellness programs.",
    10: "Plan for flu/COVID seasons—improve air quality, vaccination, and remote options.",
    11: "Allow flexible scheduling and promote nutrition/wellness resources.",
    12: "Reduce irritant exposure and ensure PPE/dermatology coverage where needed.",
    13: "Invest in ergonomics, lifting training, and access to physiotherapy.",
    14: "Provide flexible scheduling and ensure access to urology/gynecology care.",
    15: "Strengthen parental leave, prenatal accommodations, and flexible schedules.",
    16: "Offer caregiver leave/flex time where perinatal-related care is needed.",
    17: "Provide consistent medical accommodations and appointment flexibility.",
    18: "Encourage early care via telemedicine and easy appointment access.",
    19: "Tighten safety training, PPE usage, and incident root-cause reviews.",
    20: "Increase safety awareness for travel/out-of-work risks and provide guidance.",
    21: "Promote preventive care and wellness incentives to reduce avoidable absences.",
    22: "Facilitate time-off for follow-ups and streamline scheduling around shifts.",
    23: "Encourage preventive visits off-peak or offer on-site/near-site clinics.",
    24: "Schedule on-site blood drives in low-demand windows and offer paid time.",
    25: "Provide flexible time for lab work and coordinate early/late appointments.",
    26: "Reinforce attendance policy with coaching and early-warning interventions.",
    27: "Support therapy attendance with flex time and ergonomic adjustments.",
    28: "Offer dental coverage, on-site events, and scheduling flexibility.",
}
DEFAULT_RECOMMENDATION = "Review absence patterns and employee support policies."

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
    current_absences = AbsenceLog.objects.filter(status='ABSENT').order_by('-date_logged')

    # Aggregate total hours by reason (all time) with reason code + description
    reason_query = (
        AbsenceLog.objects
        .values('reason__reason_code', 'reason__description')
        .annotate(total_hours=Sum('predicted_hours'))
        .filter(total_hours__gt=0)
        .order_by('-total_hours')
    )

    chart_labels = [item['reason__description'] for item in reason_query]
    chart_values = [item['total_hours'] for item in reason_query]

    # Determine the top reason and map to recommendation
    if reason_query:
        top = reason_query[0]
        top_reason_code = top['reason__reason_code']
        top_reason_label = top['reason__description']
        top_reason_recommendation = REASON_RECOMMENDATIONS.get(top_reason_code, DEFAULT_RECOMMENDATION)
    else:
        top_reason_label = None
        top_reason_recommendation = DEFAULT_RECOMMENDATION

    # --- KPI CARDS (Current Month) ---
    current_month = timezone.now().month
    current_year = timezone.now().year

    month_logs = (
        AbsenceLog.objects
        .filter(date_logged__month=current_month, date_logged__year=current_year)
        .select_related('employee')
    )

    total_predicted_hours = month_logs.aggregate(total=Sum('predicted_hours'))['total'] or 0.0

    # Absenteeism rate = total predicted hours / (employees * 160) * 100
    STANDARD_WORK_HOURS = 160.0
    employee_count = Employee.objects.count()
    total_standard_hours = employee_count * STANDARD_WORK_HOURS
    if total_standard_hours > 0:
        absenteeism_rate = (float(total_predicted_hours) / float(total_standard_hours)) * 100.0
    else:
        absenteeism_rate = 0.0

    # Estimated compensation reduction = sum(predicted_hours * hourly_rate) across month logs
    estimated_compensation_impact = 0.0
    for log in month_logs:
        try:
            estimated_compensation_impact += float(log.predicted_hours) * float(log.employee.hourly_rate)
        except Exception:
            # In case of missing employee/hourly_rate, skip that log
            continue

    context = {
        'current_absences': current_absences,
        'chart_labels': json.dumps(chart_labels),
        'chart_data': json.dumps(chart_values),
        'top_reason_label': top_reason_label,
        'top_reason_recommendation': top_reason_recommendation,
        # KPI cards
        'overall_absenteeism_rate': f"{absenteeism_rate:.2f}%",
        'estimated_compensation_impact': round(estimated_compensation_impact, 2),
        'total_predicted_hours': round(float(total_predicted_hours), 1),
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
                X_pred
            )[0]

            predicted_hours =  round(float(predicted_hours), 2) # Round to 2 decimal places

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

    # Company-wide totals
    total_company_hours_lost = 0.0
    total_company_cost_impact = 0.0

    for emp in employees:
        absent_hours_agg = AbsenceLog.objects.filter(
            employee=emp,
            date_logged__month=current_month,
            date_logged__year=current_year
        ).aggregate(total_hours=Sum('predicted_hours'))
        
        total_absent_hours = absent_hours_agg['total_hours'] or 0.0
        # Individual calculations
        expected_work_hours = max(0, STANDARD_WORK_HOURS - float(total_absent_hours))
        absence_cost = float(total_absent_hours) * float(emp.hourly_rate)
        expected_compensation = expected_work_hours * float(emp.hourly_rate)

        # Update company-wide totals
        total_company_hours_lost += float(total_absent_hours)
        total_company_cost_impact += absence_cost
        
        salary_data.append({
            'name': emp.full_name,
            'total_absent_hours': total_absent_hours, # Pass the full float
            'expected_work_hours': round(expected_work_hours, 1),
            'absence_cost': round(absence_cost, 2),
            'expected_compensation': round(expected_compensation, 2)
        })

    context = {
        'salary_data': salary_data,
        'total_company_hours_lost': round(total_company_hours_lost, 1),
        'total_company_cost_impact': round(total_company_cost_impact, 2),
        'page': 'salaries'
    }
    return render(request, 'tracker/3_salaries.html', context)

# --- Tab 4: About the Model View (No Change) ---
def about_model_view(request):
    context = {'page': 'about'}
    return render(request, 'tracker/4_about.html', context)


def model_explanations_view(request):
    """Render a simple model explainability page.

    This page provides a human-friendly explanation of the model, lists
    top features, and reserves space for visual explainability artifacts
    (SHAP, feature importances). Keep content lightweight to avoid
    introducing heavy dependencies at render time.
    """
    # Small, safe defaults — replace or extend with live data if available
    top_features = [
        'Body mass index',
        'Service time',
        'Age',
        'Reason for absence',
        'Work load Average/day '
    ]

    context = {
        'page': 'model_explanations',
        'top_features': top_features,
        'shap_available': False,  # flip to True and pass SHAP payload when available
    }
    return render(request, 'tracker/5_model_explanations.html', context)