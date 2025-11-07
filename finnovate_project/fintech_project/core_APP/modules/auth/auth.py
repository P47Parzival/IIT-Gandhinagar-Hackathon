import logging
from django.shortcuts import render
from django.contrib.auth import authenticate, login
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth import logout


def auth_view(request):
    """Auth view."""
    if request.user.is_authenticated:
        return redirect("dashboard_page")
    return render(request, 'auth/auth.html')


def authenticate_user(request):
    """Authenticate user credentials."""
    if request.method == "POST":
        print("Authentication attempt")
        username = request.POST.get("username")
        password = request.POST.get("password")
        print(f"Received credentials - Username: {username}, Password: {password}")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            print("Authentication successful")
            login(request, user)
            return redirect("dashboard_page")
        else:
            print("Authentication failed")
            messages.error(request, "Invalid username or password.")
            return render(request, "auth/auth.html", status=401)

    return render(request, "auth/auth.html")


def logout_user(request):
    """Logout user."""
    if request.user.is_authenticated:
        logout(request)
    return redirect("auth_view")
