from django.urls import path
from .home import (
    home_view
)

urlpatterns = [
    path('', home_view, name='landing_page'),
    # path("reset_password/", reset_password_view, name="reset_password"),
    
    # path('login/', login_view, name='login'),
    # path("verify_reset_otp/", verify_reset_otp, name="verify_reset_otp"),
    # path("change_password/", change_password, name="change_password"),
]
