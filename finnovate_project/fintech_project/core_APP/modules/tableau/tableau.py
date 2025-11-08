import logging
import os
from django.shortcuts import render
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required


@login_required
def tableau_view(request):
    """Link Data view."""
    context = {}
    return render(request, 'tableau/tableau.html', context=context)