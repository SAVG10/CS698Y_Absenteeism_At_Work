# tracker/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # Tab 1: Landing Dashboard (login is default)
    path('', views.login_view, name='login'),
    # Dashboard route (used by redirects after login)
    path('dashboard/', views.dashboard_view, name='dashboard'),

    # Tab 2: Log New Absence
    path('log_absence/', views.log_absence_view, name='log_absence'),

    # Tab 3: Salaries
    path('salaries/', views.salaries_view, name='salaries'),

    # Tab 4: About the Model
    path('about/', views.about_model_view, name='about_model'),

    # Add routes for login and password reset
    path('login/', views.login_view, name='login'),
    path('password_reset/', views.password_reset_view, name='password_reset'),
]