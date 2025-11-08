from django.urls import path

from .department_management import department_management_view


urlpatterns = [
    path("", department_management_view, name="department_management_page"),
]

