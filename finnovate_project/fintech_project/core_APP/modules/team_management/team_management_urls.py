from django.urls import path

from .team_management import team_management_view


urlpatterns = [
    path("", team_management_view, name="team_management_page"),
]

