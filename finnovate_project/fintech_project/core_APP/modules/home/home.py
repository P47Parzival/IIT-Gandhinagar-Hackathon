import logging
from django.shortcuts import render


def home_view(request):
    """Home view."""
    return render(request, 'home/home.html')
