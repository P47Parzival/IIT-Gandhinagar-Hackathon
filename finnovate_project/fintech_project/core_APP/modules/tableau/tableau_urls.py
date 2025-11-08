from django.urls import path
from .tableau import (
    tableau_view,
)

urlpatterns = [
    path('', tableau_view, name='tableau_page'),
]
