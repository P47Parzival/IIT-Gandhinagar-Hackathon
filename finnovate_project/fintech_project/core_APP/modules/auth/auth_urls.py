from django.urls import path
from .auth import (
    auth_view,
    authenticate_user,
    logout_user,
)

urlpatterns = [
    path('', auth_view, name='auth_view'),
    path('login/', authenticate_user, name='auth_login'),
    path('logout/', logout_user, name='auth_logout'),
]
