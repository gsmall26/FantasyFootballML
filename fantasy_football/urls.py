# fantasy_football/urls.py
from django.urls import path
from . import views  # Import views from the same directory

urlpatterns = [
    path('', views.home, name='home'),  # URL pattern for the home page
    path('players/', views.players, name='players'),  # URL pattern for the players page
    path('draft/', views.draft, name='draft'),  # URL pattern for the draft page
]
