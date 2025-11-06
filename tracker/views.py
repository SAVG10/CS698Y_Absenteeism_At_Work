# tracker/views.py

from django.shortcuts import render, redirect
from django.db.models import Count, Sum, Q
from django.utils import timezone
from .models import Employee, AbsenceReason, AbsenceLog, EmployeePassword
import joblib
import pandas as pd
import json
from django.contrib.auth.hashers import check_password, make_password

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
        'page': 'dashboard',
    }

    # --- Model accuracy metrics (based on logs where actual_hours is available) ---
    logs_with_actual = AbsenceLog.objects.filter(actual_hours__isnull=False)
    total_actual = logs_with_actual.count()
    if total_actual > 0:
        abs_errors = []
        within_tol = 0
        TOLERANCE_HOURS = 1.0  # Consider a prediction "accurate" if within +/- 1 hour
        for log in logs_with_actual:
            try:
                pred = float(log.predicted_hours)
                actual = float(log.actual_hours)
            except Exception:
                # Skip malformed entries
                continue
            diff = abs(pred - actual)
            abs_errors.append(diff)
            if diff <= TOLERANCE_HOURS:
                within_tol += 1

        # Mean Absolute Error (hours)
        model_mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        # Accuracy = percent of predictions within tolerance
        model_accuracy_pct = round((within_tol / total_actual) * 100.0)

        context['model_accuracy'] = f"{model_accuracy_pct}%"
        context['model_mae'] = round(model_mae, 2)
    else:
        context['model_accuracy'] = 'N/A'
        context['model_mae'] = None
    return render(request, 'tracker/1_dashboard.html', context)

# --- Tab 2: Log Absence View (THE FINAL FIX) ---
def log_absence_view(request):
    # Base context used by the template
    context = {
        'employees': Employee.objects.all(),
        'reasons': AbsenceReason.objects.all(),
        'prediction_result': None,
        'page': 'log_absence'
    }

    # Provide unresolved absences for the "real" mode (those that are still ABSENT and have no actual_hours)
    context['unresolved_absences'] = (
        AbsenceLog.objects
        .filter(status='ABSENT', actual_hours__isnull=True)
        .select_related('employee', 'reason')
        .order_by('-date_logged')
    )

    # Handle POST actions: 'predict', 'log' (confirm & save prediction), and 'add_actual' (real data)
    if request.method == 'POST':
        action = request.POST.get('action', 'predict')

        # === Add actual hours for an existing predicted absence ===
        if action == 'add_actual':
            try:
                log_id = int(request.POST.get('absence_log_id'))
                actual_val = float(request.POST.get('actual_hours_taken'))
                log = AbsenceLog.objects.get(pk=log_id)
                log.actual_hours = actual_val
                # Mark as returned when actual hours are provided
                log.status = 'RETURNED'
                log.save()
                context['actual_saved'] = {
                    'name': log.employee.full_name,
                    'hours': actual_val
                }
            except Exception as e:
                context['error'] = f"Failed to save actual hours: {e}"
                print(f"--- ERROR saving actual hours: {e} ---")

        # === Prediction and logging flow (requires model) ===
        elif action in ('predict', 'log'):
            if not TO_BMI_MODEL:
                context['error'] = 'Prediction model unavailable. Cannot run prediction.'
            else:
                try:
                    # Collect inputs
                    employee_id = int(request.POST.get('employee_id'))
                    reason_code = int(request.POST.get('reason_code'))

                    employee = Employee.objects.get(employee_id=employee_id)
                    reason = AbsenceReason.objects.get(reason_code=reason_code)

                    feature_dict = {
                        'Reason for absence': reason_code,
                        'Month of absence': timezone.now().month,
                        'Day of the week': timezone.now().isoweekday() + 1,
                        'Seasons': (timezone.now().month%12 + 3)//3,
                        'Transportation expense': employee.transportation_expense,
                        'Distance from Residence to Work': employee.distance_from_residence_to_work,
                        'Service time': employee.service_time,
                        'Age': employee.age,
                        'Work load Average/day ': employee.work_load_average_day, 
                        'Hit target': employee.hit_target,
                        'Body mass index': employee.body_mass_index,
                    }

                    X_pred = pd.DataFrame([feature_dict])

                    predicted_hours = TO_BMI_MODEL.predict(X_pred)[0]
                    predicted_hours = round(float(predicted_hours), 2)

                    # Expose values so the template can show the prediction and let the user confirm
                    context['predicted_hours'] = predicted_hours
                    context['selected_employee_id'] = str(employee_id)
                    context['selected_reason_code'] = str(reason_code)

                    # If this POST is the final 'log' action, save the record (allow override via actual_hours input)
                    if action == 'log':
                        save_hours = predicted_hours
                        override = request.POST.get('actual_hours')
                        if override:
                            try:
                                save_hours = float(override)
                            except Exception:
                                pass

                        AbsenceLog.objects.create(
                            employee=employee,
                            reason=reason,
                            predicted_hours=save_hours,
                            status='ABSENT'
                        )
                        context['prediction_result'] = {
                            'name': employee.full_name,
                            'hours': save_hours
                        }
                    else:
                        # action == 'predict' -> user will be shown the prediction and can confirm
                        pass

                except Exception as e:
                    context['error'] = f"Prediction failed: {e}. Check terminal for details."
                    print(f"--- PREDICTION ERROR: {e} ---")

    return render(request, 'tracker/2_log_absence.html', context)

# --- Tab 3: Salaries View (No Change) ---
def salaries_view(request):
    STANDARD_WORK_HOURS = 160.0
    # Read filter params from GET
    search = request.GET.get('search', '').strip()
    severity = request.GET.get('severity', '').strip()  # low|medium|high or ''

    # Start with all employees, then apply search filtering at the queryset level where possible
    employees = Employee.objects.all()
    if search:
        # If search is numeric, try matching employee_id; always also match name (case-insensitive)
        try:
            emp_id_val = int(search)
        except Exception:
            emp_id_val = None

        if emp_id_val is not None:
            # If the search is numeric, match employee_id exactly (do not match name substrings)
            employees = employees.filter(employee_id=emp_id_val)
        else:
            # Non-numeric search: match on name (case-insensitive)
            employees = employees.filter(full_name__icontains=search)
    salary_data = []
    
    current_month = timezone.now().month
    current_year = timezone.now().year

    # Compute overall (unfiltered) company totals for the current month so the header shows company-wide stats
    overall_logs = AbsenceLog.objects.filter(date_logged__month=current_month, date_logged__year=current_year).select_related('employee')
    overall_company_hours_lost = 0.0
    overall_company_cost_impact = 0.0
    for log in overall_logs:
        try:
            overall_company_hours_lost += float(log.predicted_hours)
            overall_company_cost_impact += float(log.predicted_hours) * float(log.employee.hourly_rate)
        except Exception:
            continue

    # Company-wide totals (used for table/footer are computed over the filtered set below)
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
            'employee_id': emp.employee_id,

            'total_absent_hours': total_absent_hours, # Pass the full float
            'expected_work_hours': round(expected_work_hours, 1),
            'absence_cost': round(absence_cost, 2),
            'expected_compensation': round(expected_compensation, 2)
        })
    # Apply severity filter to the built salary_data list (severity depends on aggregated hours)
    if severity in ('low', 'medium', 'high'):
        def severity_ok(item):
            h = float(item['total_absent_hours'] or 0.0)
            if severity == 'low':
                return 0.0 <= h <= 4.0
            if severity == 'medium':
                return 4.0 < h <= 8.0
            if severity == 'high':
                return h > 8.0
            return True

        filtered_salary_data = [itm for itm in salary_data if severity_ok(itm)]
    else:
        filtered_salary_data = salary_data

    # Recompute totals for the filtered set
    total_company_hours_lost = sum(float(it['total_absent_hours'] or 0.0) for it in filtered_salary_data)
    total_company_cost_impact = sum(float(it['absence_cost'] or 0.0) for it in filtered_salary_data)

    context = {
        'salary_data': filtered_salary_data,
        'total_company_hours_lost': round(total_company_hours_lost, 1),
        'total_company_cost_impact': round(total_company_cost_impact, 2),
        'overall_company_hours_lost': round(overall_company_hours_lost, 1),
        'overall_company_cost_impact': round(overall_company_cost_impact, 2),
        'page': 'salaries',
        'search': search,
        'severity': severity,
    }
    return render(request, 'tracker/3_salaries.html', context)

# --- Tab 4: About the Model View (No Change) ---
def about_model_view(request):
    context = {'page': 'about'}
    return render(request, 'tracker/4_about.html', context)

# --- Login View ---
def login_view(request):
    if request.method == 'POST':
        login_type = request.POST.get('login_type')  # 'admin' or 'employee'

        if login_type == 'admin':
            admin_password = request.POST.get('password')
            if admin_password == 'admin':
                # Redirect to dashboard for admin
                return redirect('dashboard')  # Corrected URL pattern name
            else:
                error_message = 'Invalid admin password.'
                return render(request, 'tracker/login.html', {'error_message': error_message})

        elif login_type == 'employee':
            employee_id = request.POST.get('employee_id')
            password = request.POST.get('password')

            try:
                user = EmployeePassword.objects.get(username=employee_id)
                if check_password(password, user.password):
                    # Redirect to dashboard for employee
                    return redirect('dashboard')  # Corrected URL pattern name
                else:
                    error_message = 'Invalid employee password.'
            except EmployeePassword.DoesNotExist:
                error_message = 'Employee does not exist.'

            return render(request, 'tracker/login.html', {'error_message': error_message})

    return render(request, 'tracker/login.html')

# --- Password Reset View ---
def password_reset_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        new_password = request.POST.get('new_password')

        try:
            user = EmployeePassword.objects.get(username=username)
            user.password = make_password(new_password)
            user.save()
            success_message = 'Password reset successfully.'
            return render(request, 'tracker/password_reset.html', {'success_message': success_message})
        except EmployeePassword.DoesNotExist:
            error_message = 'User does not exist.'
            return render(request, 'tracker/password_reset.html', {'error_message': error_message})

    return render(request, 'tracker/password_reset.html')