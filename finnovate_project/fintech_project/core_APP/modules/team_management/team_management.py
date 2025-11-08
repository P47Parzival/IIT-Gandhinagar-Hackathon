from django import forms
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError, transaction
from django.db.models import Q
from django.http import HttpResponseForbidden
from django.shortcuts import redirect, render
from django.conf import settings
from django.core.mail import send_mail


from core_APP.models import CustomUser, ResponsibilityMatrix, BalanceSheet


class AddTeamMemberForm(forms.Form):
    ROLE_CHOICES = (
        ("4", "Preparer"),
        ("5", "Reviewer"),
    )

    user = forms.ModelChoiceField(
        queryset=CustomUser.objects.filter(user_type=4),
        label="Select User",
        help_text="Choose a user to add to your team.",
    )
    user_role = forms.ChoiceField(
        choices=ROLE_CHOICES,
        label="Assign Role",
        help_text="Select Preparer/Maker or Reviewer role.",
    )


class AssignGLCodeForm(forms.Form):
    user_id = forms.ChoiceField(
        label="Select User",
        choices=[],
    )
    gl_code = forms.ChoiceField(
        label="Select GL Code",
        choices=[],
    )

    def __init__(self, *args, **kwargs):
        team_members = kwargs.pop('team_members', None)
        gl_codes_list = kwargs.pop('gl_codes_list', None)
        super().__init__(*args, **kwargs)
        
        if team_members:
            # Create choices from team members
            user_choices = [
                (str(member['user'].id), f"{member['user'].get_full_name() or member['user'].username}: {member['user'].email}")
                for member in team_members
            ]
            self.fields['user_id'].choices = [('', '---------')] + user_choices
        
        if gl_codes_list:
            self.fields['gl_code'].choices = [('', '---------')] + gl_codes_list


@login_required
def team_management_view(request):
    # Only allow Department Heads (user_type 2) to access
    if getattr(request.user, "user_type", None) != 2:
        return HttpResponseForbidden("You do not have permission to access this page.")

    # Get the current user's department from their ResponsibilityMatrix entry
    try:
        user_responsibility = ResponsibilityMatrix.objects.filter(
            user=request.user, user_role=2
        ).first()
        if not user_responsibility or not user_responsibility.department:
            messages.error(
                request,
                "You are not assigned to a department. Please contact an administrator.",
            )
            return redirect("dashboard_page")
        department = user_responsibility.department
    except Exception:
        messages.error(
            request,
            "Unable to retrieve your department information. Please contact an administrator.",
        )
        return redirect("dashboard_page")

   # Get unique team members from ResponsibilityMatrix for this department
    responsibility_entries = (
        ResponsibilityMatrix.objects.filter(department=department)
        .exclude(user=request.user)
        .select_related("user")
    )

    # Get unique user IDs (SQLite-safe)
    unique_user_ids = {entry.user_id for entry in responsibility_entries}

    # Get the users and annotate with their role and GL codes
    team_members = []
    for user_id in unique_user_ids:
        user_entries = responsibility_entries.filter(user_id=user_id)
        resp_entry = user_entries.first()
        if resp_entry:
            gl_codes = user_entries.filter(gl_code__isnull=False).values_list("gl_code", flat=True)
            team_members.append({
                'user': resp_entry.user,
                'user_role': resp_entry.get_user_role_display(),
                'user_role_code': resp_entry.user_role,
                'gl_codes': list(gl_codes) if gl_codes else [],
            })

    # Sort by name
    team_members.sort(key=lambda x: (x['user'].first_name or '', x['user'].last_name or ''))


    # Get available users (user_type 4) that are not already in the team
    existing_user_ids = [member['user'].id for member in team_members]
    available_users = CustomUser.objects.filter(user_type=4).exclude(
        id__in=existing_user_ids
    )
    
    # Handle search query for available users
    search_query = request.GET.get("search", "").strip()
    if search_query:
        available_users = available_users.filter(
            Q(first_name__icontains=search_query) |
            Q(last_name__icontains=search_query) |
            Q(username__icontains=search_query) |
            Q(email__icontains=search_query)
        )
    
    # Prepare GL codes list for the form (from all users in the department's balance sheets)
    gl_codes_list = []
    
    # Get unique GL codes from all balance sheets in this department
    unique_gl_codes = BalanceSheet.objects.filter(
        responsible_department=department
    ).values_list('gl_acct', flat=True).distinct()
    
    for gl_code in unique_gl_codes:
        # Get the first BalanceSheet entry for this GL code in the department
        bs = BalanceSheet.objects.filter(
            responsible_department=department,
            gl_acct=gl_code
        ).first()
        if bs:
            display_name = f"{bs.gl_acct} - {bs.gl_account_name or 'N/A'}"
            gl_codes_list.append((bs.gl_acct, display_name))
    
    # Sort by GL code
    gl_codes_list.sort(key=lambda x: x[0])

    # Handle form submissions
    if request.method == "POST":
        form_type = request.POST.get('form_type')
        
        if form_type == 'add_member':
            form = AddTeamMemberForm(request.POST)
            form.fields["user"].queryset = available_users
            gl_form = AssignGLCodeForm(team_members=team_members, gl_codes_list=gl_codes_list)

            if form.is_valid():
                selected_user = form.cleaned_data["user"]
                user_role = int(form.cleaned_data["user_role"])

                try:
                    with transaction.atomic():
                        ResponsibilityMatrix.objects.create(
                            user=selected_user,
                            department=department,
                            user_role=user_role,
                            gl_code=None,
                            gl_code_status=None,
                        )

                    messages.success(
                        request,
                        f"User '{selected_user.get_full_name() or selected_user.username}' added to team successfully.",
                    )
                except IntegrityError:
                    form.add_error(
                        None, "Unable to add user to team. They may already be in the team."
                    )
                
                # Welcome Email
                try:
                    send_mail(
                        subject="Hello from Finnovate Project 2025",
                        message=f"Hi {selected_user.first_name},\n\nWelcome to Finnovate Project 2025! You were added by {request.user.first_name} {request.user.last_name} from {department} Department.\nWe are excited to have you on board.\n\nBest regards,\nFinnovate Project 2025",
                        from_email=settings.DEFAULT_FROM_EMAIL,
                        recipient_list=[selected_user.email],
                        fail_silently=False,
                    )
                except Exception as e:
                    print(f"Error sending welcome email: {e}")

                return redirect("team_management_page")
            else:
                messages.error(request, "Please correct the errors below and resubmit.")
        
        elif form_type == 'assign_gl':
            gl_form = AssignGLCodeForm(request.POST, team_members=team_members, gl_codes_list=gl_codes_list)
            form = AddTeamMemberForm()
            form.fields["user"].queryset = available_users

            if gl_form.is_valid():
                selected_user_id = gl_form.cleaned_data["user_id"]
                selected_gl_code = gl_form.cleaned_data["gl_code"]
                
                try:
                    # Get the user object
                    selected_user = CustomUser.objects.get(id=selected_user_id)
                    
                    # Check if user already has this GL code assigned
                    existing_gl_entry = ResponsibilityMatrix.objects.filter(
                        user=selected_user,
                        gl_code=selected_gl_code
                    ).first()
                    
                    if existing_gl_entry:
                        messages.error(request, f"This GL code is already assigned to {selected_user.get_full_name() or selected_user.username}.")
                        return redirect("team_management_page")
                    
                    with transaction.atomic():
                        # Check if user has a record with no gl_code in this department
                        existing_entry = ResponsibilityMatrix.objects.filter(
                            user=selected_user,
                            department=department,
                            gl_code__isnull=True
                        ).first()
                        
                        if existing_entry:
                            # Update existing record
                            existing_entry.gl_code = selected_gl_code
                            existing_entry.gl_code_status = 1  # Pending
                            existing_entry.save()
                        else:
                            # Create new record with gl_code
                            # Get the user's role from their existing entries
                            user_role_entry = ResponsibilityMatrix.objects.filter(
                                user=selected_user,
                                department=department
                            ).first()
                            
                            if user_role_entry:
                                ResponsibilityMatrix.objects.create(
                                    user=selected_user,
                                    department=department,
                                    user_role=user_role_entry.user_role,
                                    gl_code=selected_gl_code,
                                    gl_code_status=1,  # Pending
                                )
                            else:
                                messages.error(request, "Unable to find user role for assignment.")
                                return redirect("team_management_page")
                    
                    messages.success(
                        request,
                        f"GL Code '{selected_gl_code}' assigned to {selected_user.get_full_name() or selected_user.username} successfully.",
                    )

                    # Info Email
                    try:
                        # Get GL Name (if exists)
                        gl_name = BalanceSheet.objects.filter(gl_acct=selected_gl_code).values_list("gl_account_name", flat=True).first() or "N/A"

                        subject = f"General Ledger Review  — {selected_gl_code}:({gl_name})"
                        message = f"""
                        Hi {selected_user.first_name},
                        You’ve been assigned a new General Ledger (GL) Code in the {department.name} department.

                        Here are the details:

                        GL Code: {selected_gl_code}
                        GL Name: {gl_name}
                        GL Review Status: Pending

                        If you believe this was assigned in error, please contact {department} Department Head/SPOC.

                        Best regards,
                        Finnovate Project 2025
                        """
                        # Send the email
                        send_mail(
                            subject=subject,
                            message=message.strip(),
                            from_email=settings.DEFAULT_FROM_EMAIL,
                            recipient_list=[selected_user.email],
                            fail_silently=False,
                        )
                    except Exception as e:
                        print(f"Error sending welcome email: {e}")
                        
                    return redirect("team_management_page")
                except CustomUser.DoesNotExist:
                    messages.error(request, "Selected user not found.")
                except Exception as e:
                    messages.error(request, f"Error assigning GL code: {str(e)}")
            else:
                messages.error(request, "Please correct the errors in the GL assignment form.")
    else:
        form = AddTeamMemberForm()
        form.fields["user"].queryset = available_users
        gl_form = AssignGLCodeForm(team_members=team_members, gl_codes_list=gl_codes_list)

    context = {
        "form": form,
        "gl_form": gl_form,
        "team_members": team_members,
        "department": department,
        "available_users": available_users,
        "search_query": search_query,
    }
    return render(request, "team_management/team_management.html", context)