# tracker/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # Tab 1: Landing Dashboard
    path('', views.dashboard_view, name='dashboard'),

    # Tab 2: Log New Absence
    path('log_absence/', views.log_absence_view, name='log_absence'),

    # Tab 3: Salaries
    path('salaries/', views.salaries_view, name='salaries'),

    # Tab 4: About the Model
    path('about/', views.about_model_view, name='about_model'),
    # Model Explainations (new)
    path('model_explanations/', views.model_explanations_view, name='model_explanations'),
]