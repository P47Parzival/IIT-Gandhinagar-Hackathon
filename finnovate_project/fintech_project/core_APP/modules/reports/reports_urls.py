from django.urls import path

from .reports import reports_view


urlpatterns = [
    path('', reports_view, name='reports_page'),
]

