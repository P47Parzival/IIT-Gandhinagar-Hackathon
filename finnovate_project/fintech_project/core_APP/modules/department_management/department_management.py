from django import forms
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.http import HttpResponseForbidden
from django.shortcuts import redirect, render

from core_APP.models import CustomUser, Department, ResponsibilityMatrix, BalanceSheet


class DepartmentUserForm(forms.Form):
    ROLE_CHOICES = (
        ("2", "Department Head"),
        ("3", "BUFC"),
    )

    first_name = forms.CharField(max_length=150, label="First Name", required=True)
    last_name = forms.CharField(max_length=150, label="Last Name", required=True)
    username = forms.CharField(max_length=150, label="Username")
    email = forms.EmailField(label="Email")
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput,
    )
    user_role = forms.ChoiceField(
        choices=ROLE_CHOICES,
        label="Assign Role",
    )
    department = forms.ModelChoiceField(
        queryset=Department.objects.all(),
        label="Assign Department",
    )

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        if CustomUser.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError("A user with this username already exists.")
        return username

    def clean_email(self):
        email = self.cleaned_data["email"].strip()
        if CustomUser.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("A user with this email already exists.")
        return email


@login_required
def department_management_view(request):
    if getattr(request.user, "user_type", None) != 1:
        return HttpResponseForbidden("You do not have permission to access this page.")
    
    context = {}
    search_query = request.GET.get("q", "").strip()
    departments = Department.objects.all()

    for dept in departments:
        spocs = (
            BalanceSheet.objects
            .filter(responsible_department=dept.name)  # exact match
            .exclude(department_spoc__isnull=True)
            .exclude(department_spoc__exact="")
            .values_list("department_spoc", flat=True)
            .distinct()
        )

        dept.department_spoc = ", ".join(sorted(spocs)) if spocs else None


    if search_query:
        departments = departments.filter(name__icontains=search_query)

    if request.method == "POST":
        form = DepartmentUserForm(request.POST)
        if form.is_valid():
            first_name = form.cleaned_data["first_name"]
            last_name = form.cleaned_data["last_name"]
            username = form.cleaned_data["username"]
            email = form.cleaned_data["email"]
            password = form.cleaned_data["password"]
            user_role = int(form.cleaned_data["user_role"])
            department = form.cleaned_data["department"]  # Already a Department instance from ModelChoiceField

            try:
                with transaction.atomic():
                    user = CustomUser.objects.create_user(
                        username=username,
                        email=email,
                        password=password,
                        first_name=first_name,
                        last_name=last_name,
                        user_type=user_role,
                    )

                    ResponsibilityMatrix.objects.create(
                        user=user,
                        department=department,
                        user_role=user_role,
                        gl_code=None,
                        gl_code_status=None,
                    )

                messages.success(
                    request,
                    f"User '{username}' created successfully and responsibility recorded.",
                )
                return redirect("department_management_page")
            except IntegrityError:
                form.add_error(None, "Unable to create user. Please try again.")
        else:
            messages.error(request, "Please correct the errors below and resubmit.")
    else:
        form = DepartmentUserForm()

    context = {
        "form": form,
        "departments": departments,
        "search_query": search_query,
    }
    return render(request, "department_management/department_management.html", context)

