# tracker/admin.py

from django.contrib import admin
from .models import AbsenceReason, Employee, AbsenceLog

class EmployeeAdmin(admin.ModelAdmin):
    list_display = ('employee_id', 'full_name', 'age', 'education', 'service_time', 'hourly_rate')

class AbsenceReasonAdmin(admin.ModelAdmin):
    list_display = ('reason_code', 'description')

class AbsenceLogAdmin(admin.ModelAdmin):
    list_display = ('employee', 'reason', 'predicted_hours', 'status', 'date_logged')
    list_filter = ('status', 'date_logged')

admin.site.register(Employee, EmployeeAdmin)
admin.site.register(AbsenceReason, AbsenceReasonAdmin)
admin.site.register(AbsenceLog, AbsenceLogAdmin)