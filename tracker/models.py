# tracker/models.py

from django.db import models
from django.utils import timezone

class AbsenceReason(models.Model):
    """ Stores the 28 reasons for absence from the dataset. """
    reason_code = models.IntegerField(primary_key=True, help_text="e.g., 1-28")
    description = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.reason_code}: {self.description}"

class Employee(models.Model):
    """ 
    Stores STATIC employee data (personal details) 
    needed for predictions and salary. 
    """
    # Basic Info
    employee_id = models.IntegerField(primary_key=True, unique=True)
    full_name = models.CharField(max_length=100)
    hourly_rate = models.DecimalField(max_digits=6, decimal_places=2, default=30.00)

    # Static features for the 'FAIR' model
    # Features REMOVED: Reason, Month, Day, Season
    transportation_expense = models.IntegerField()
    distance_from_residence_to_work = models.IntegerField()
    service_time = models.IntegerField()
    age = models.IntegerField()
    work_load_average_day = models.FloatField()
    hit_target = models.IntegerField()
    education = models.IntegerField(help_text="1:High School, 2:Graduate, 3:Postgrad, 4:Doctor")
    body_mass_index = models.FloatField()

    def __str__(self):
        return self.full_name

class AbsenceLog(models.Model):
    """ 
    Logs all reported absence EVENTS and stores the model's prediction.
    This is the "different table" you mentioned.
    """
    STATUS_CHOICES = [
        ('ABSENT', 'Currently Absent'),
        ('RETURNED', 'Returned to Work'),
    ]

    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    reason = models.ForeignKey(AbsenceReason, on_delete=models.PROTECT)
    date_logged = models.DateField(default=timezone.now)
    predicted_hours = models.FloatField()  # <-- The model prediction is stored here
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='ABSENT')

    def __str__(self):
        return f"{self.employee.full_name} - {self.reason.description}"