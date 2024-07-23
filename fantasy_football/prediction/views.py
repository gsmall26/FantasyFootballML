# fantasy_football/views.py
from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def players(request):
    return render(request, 'players.html')

def draft(request):
    return render(request, 'draft.html')
