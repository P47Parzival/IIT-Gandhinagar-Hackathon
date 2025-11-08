from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import transaction
from django.views.decorators.http import require_http_methods
import logging
from core_APP.models import BalanceSheet, ResponsibilityMatrix, TrialBalance, GLReview, GLSupportingDocument,   ReviewTrail
from django.utils import timezone

logger = logging.getLogger(__name__)


# @login_required
# def reports_view(request):
#     """Static reports landing page."""

#     # Static data mirroring the RACI-style matrix shown in the reference image
#     roles = [
#         {"label": "Project Lead", "name": "Anne"},
#         {"label": "Internal Recruiter", "name": "John"},
#         {"label": "Hiring Manager", "name": "Natasha"},
#         {"label": "Stakeholder 4", "name": "Steven"},
#         {"label": "Stakeholder 5", "name": "Sarah"},
#         {"label": "Stakeholder 8", "name": "Allison"},
#     ]

#     raw_tasks = [
#         {
#             "title": "Task 1",
#             "caption": "Defining the job role",
#             "assignments": ["A", "A", "R", "I", "R", "A"],
#         },
#         {
#             "title": "Task 2",
#             "caption": "Creating a requisition",
#             "assignments": ["A", "R", "I", "C", "I", "A"],
#         },
#         {
#             "title": "Task 3",
#             "caption": "Writing the job ad",
#             "assignments": ["C", "A", "C", "A", "C", "C"],
#         },
#         {
#             "title": "Task 4",
#             "caption": "Posting the job ad",
#             "assignments": ["C", "R", "I", "R", "I", "C"],
#         },
#         {
#             "title": "Task 5",
#             "caption": "Promote the position on the company channels",
#             "assignments": ["C", "A", "I", "R", "I", "C"],
#         },
#         {
#             "title": "Task 6",
#             "caption": "Advertise the position internally",
#             "assignments": ["I", "A", "R", "C", "R", "I"],
#         },
#         {
#             "title": "Task 7",
#             "caption": "Review applications",
#             "assignments": ["A", "I", "R", "I", "R", "A"],
#         },
#         {
#             "title": "Task 8",
#             "caption": "Candidate screening",
#             "assignments": ["C", "I", "C", "I", "C", "C"],
#         },
#     ]

#     matrix_rows = []
#     for task in raw_tasks:
#         cells = []
#         for role, assignment in zip(roles, task["assignments"]):
#             cells.append(
#                 {
#                     "assignment": assignment,
#                     "role_label": f"{role['label']} — {role['name']}",
#                 }
#             )

#         matrix_rows.append(
#             {
#                 "title": task["title"],
#                 "caption": task["caption"],
#                 "cells": cells,
#             }
#         )

#     context = {
#         "roles": roles,
#         "tasks": matrix_rows,
#     }
#     return render(request, 'reports/reports.html', context)



@login_required
def gl_reviews_view(request):
    """GL Reviews page - shows both Preparer and Reviewer GLs side by side."""

    if request.user.user_type != 4:
        messages.error(request, "You do not have permission to access this page.")
        return redirect("dashboard_page")
    
    # Get Preparer GLs (user_role = 4)
    preparer_assignments = ResponsibilityMatrix.objects.filter(
        user=request.user,
        gl_code__isnull=False,
        user_role=4
    ).select_related('department').order_by('gl_code')
    
    # Get Reviewer GLs (user_role = 5)
    reviewer_assignments = ResponsibilityMatrix.objects.filter(
        user=request.user,
        gl_code__isnull=False,
        user_role=5
    ).select_related('department').order_by('gl_code')
    
    def prepare_gl_data(assignments, is_preparer=False):
        """Helper function to prepare GL data with supporting documents."""
        gl_data = []
        for assignment in assignments:
            # Get GL details from BalanceSheet
            balance_sheet = BalanceSheet.objects.filter(
                gl_acct=assignment.gl_code
            ).first()
            
            # For Preparer: find TrialBalance by user and GL code
            # For Reviewer: find TrialBalance by GL code only (could be created by preparer)
            if is_preparer:
                trial_balance = TrialBalance.objects.filter(
                    user=request.user,
                    gl_code=assignment.gl_code
                ).first()
            else:
                # Reviewer: find any TrialBalance with this GL code
                trial_balance = TrialBalance.objects.filter(
                    gl_code=assignment.gl_code
                ).first()
            
            # Get GLReview if exists
            # For Preparer: find review by this user
            # For Reviewer: find any review for this GL code (created by preparer)
            gl_review = None
            supporting_docs = []
            if trial_balance:
                if is_preparer:
                    gl_review = GLReview.objects.filter(
                        trial_balance=trial_balance,
                        reviewer=request.user
                    ).first()
                else:
                    # Reviewer: find any GLReview for this GL code
                    gl_review = GLReview.objects.filter(
                        trial_balance=trial_balance
                    ).first()
                
                if gl_review:
                    supporting_docs = GLSupportingDocument.objects.filter(
                        gl_review=gl_review
                    ).order_by('-uploaded_at')
            
            # Get status from the assignment, but also check if there's a GLReview
            # If GLReview exists, use its status; otherwise use assignment status
            status_code = assignment.gl_code_status or 1
            status_display = assignment.get_gl_code_status_display() if assignment.gl_code_status else 'Pending'
            
            # If GLReview exists, use its status
            if gl_review:
                status_code = gl_review.status
                status_display = gl_review.get_status_display()
            
            # Format assigned date
            assigned_on = assignment.created_at
            assigned_on_formatted = assigned_on.strftime("%B %d, %Y at %I:%M %p") if assigned_on else 'N/A'
            
            gl_data.append({
                'assignment_id': str(assignment.id),
                'gl_code': assignment.gl_code,
                'gl_name': balance_sheet.gl_account_name if balance_sheet else 'N/A',
                'department': assignment.department.name if assignment.department else 'N/A',
                'user_role': assignment.get_user_role_display(),
                'status': status_display,
                'status_code': status_code,
                'assigned_on': assigned_on_formatted,
                'assigned_on_datetime': assigned_on,
                'trial_balance_id': str(trial_balance.id) if trial_balance else None,
                'gl_review_id': str(gl_review.id) if gl_review else None,
                'supporting_documents': [
                    {
                        'id': str(doc.id),
                        'file_name': doc.file.name.split('/')[-1] if doc.file else 'Unknown',
                        'file_url': doc.file.url if doc.file else '#',
                        'uploaded_at': doc.uploaded_at.strftime("%B %d, %Y at %I:%M %p") if doc.uploaded_at else 'N/A',
                    }
                    for doc in supporting_docs
                ],
            })
        return gl_data
    
    preparer_gls = prepare_gl_data(preparer_assignments, is_preparer=True)
    reviewer_gls = prepare_gl_data(reviewer_assignments, is_preparer=False)
    
    context = {
        'preparer_gls': preparer_gls,
        'reviewer_gls': reviewer_gls,
    }
    return render(request, 'gl_reviews/gl_reviews_t3.html', context)


@login_required
def upload_gl_supporting_document(request):
    """Upload a supporting document for a GL review (Preparer only)."""
    if request.method != 'POST':
        messages.error(request, "Invalid request method.")
        return redirect("gl_reviews_page")
    
    if request.user.user_type != 4:
        messages.error(request, "You do not have permission to upload documents.")
        return redirect("gl_reviews_page")
    
    assignment_id = request.POST.get('assignment_id')
    gl_code = request.POST.get('gl_code')
    uploaded_file = request.FILES.get('supporting_document')
    
    if not assignment_id or not gl_code or not uploaded_file:
        messages.error(request, "Missing required information.")
        return redirect("gl_reviews_page")
    
    try:
        # Verify the assignment belongs to the user and is Preparer role
        assignment = ResponsibilityMatrix.objects.get(
            id=assignment_id,
            user=request.user,
            gl_code=gl_code,
            user_role=4  # Preparer only
        )
        
        # Get or create TrialBalance
        trial_balance = TrialBalance.objects.filter(
            user=request.user,
            gl_code=gl_code
        ).first()
        
        if not trial_balance:
            balance_sheet = BalanceSheet.objects.filter(gl_acct=gl_code).first()
            trial_balance = TrialBalance.objects.create(
                user=request.user,
                gl_code=gl_code,
                gl_name=balance_sheet.gl_account_name if balance_sheet else '',
                amount=0,
            )
        
        # Get or create GLReview
        gl_review = GLReview.objects.filter(
            trial_balance=trial_balance,
            reviewer=request.user
        ).first()
        
        if not gl_review:
            gl_review = GLReview.objects.create(
                trial_balance=trial_balance,
                reviewer=request.user,
                reconciliation_notes='',
                status=1,  # Pending
            )
        
        # Create GLSupportingDocument
        with transaction.atomic():
            GLSupportingDocument.objects.create(
                gl_review=gl_review,
                file=uploaded_file,
            )
        
        messages.success(request, f"Supporting document uploaded successfully for GL Code {gl_code}.")
        
    except ResponsibilityMatrix.DoesNotExist:
        messages.error(request, "GL assignment not found or you don't have permission to upload documents for this GL.")
    except Exception as e:
        messages.error(request, f"Error uploading document: {str(e)}")
    
    return redirect("gl_reviews_page")


@login_required
@require_http_methods(["POST"])
def submit_gl_review_preparer(request):
    """
    Submit a GL review for an assigned GL code.
    """
    
    assignment_id = request.POST.get('assignment_id')
    gl_code = request.POST.get('gl_code')
    reconciliation_notes = request.POST.get('reconciliation_notes', '').strip()
    
    if not assignment_id or not gl_code:
        messages.error(request, "Missing required information.")
        return redirect("gl_reviews_page")
    
    if not reconciliation_notes:
        messages.error(request, "Reconciliation notes are required.")
        return redirect("gl_reviews_page")
    
    try:
        # Verify the assignment belongs to the user
        assignment = ResponsibilityMatrix.objects.get(
            id=assignment_id,
            gl_code=gl_code
        )
        
        # Find or create a TrialBalance entry for this GL code
        # We need to link GLReview to TrialBalance, so we'll find or create one
        trial_balance = TrialBalance.objects.filter(
            gl_code=gl_code
        ).first()
        
        # If no TrialBalance exists, create one (minimal entry)
        if not trial_balance:
            trial_balance = TrialBalance.objects.create(
                user=request.user,
                gl_code=gl_code,
                gl_name=BalanceSheet.objects.filter(gl_acct=gl_code).values_list('gl_account_name', flat=True).first() or '',
                amount=0,
            )
        
        # Create or update GLReview
        with transaction.atomic():
            # Check if review already exists
            gl_review = GLReview.objects.filter(
                trial_balance=trial_balance,
                reviewer=request.user
            ).first()
            
            if gl_review:
                # Update existing review
                gl_review.reconciliation_notes = reconciliation_notes
                gl_review.status = 2  # Awaiting Approval
                gl_review.save()
            else:
                # Create new review
                gl_review = GLReview.objects.create(
                    trial_balance=trial_balance,
                    reviewer=request.user,
                    reconciliation_notes=reconciliation_notes,
                    status=2,  # Awaiting Approval
                )
            
            # Update ResponsibilityMatrix status
            assignment.gl_code_status = 2  # Done
            assignment.save()

            # Also mark in ReviewTrail
            review_trail = ReviewTrail.objects.create(
                reviewer=request.user,
                reviewer_responsibility_matrix=assignment,
                gl_review=gl_review,
                previous_trail=None,
            )
            review_trail.save()
        
        messages.success(
            request,
            f"GL Review submitted successfully for GL Code {gl_code}. Status updated to 'Awaiting Approval'."
        )
        
    except ResponsibilityMatrix.DoesNotExist:
        messages.error(request, "GL assignment not found or does not belong to you.")
    except Exception as e:
        logger.error(f"Error submitting GL review: {str(e)}")
        messages.error(request, f"Error submitting review: {str(e)}")
    
    return redirect("gl_reviews_page")


@login_required
def balance_sheet_view(request):
    """Detailed balance sheet view for Tier 3 navigation."""

    sheets = (
        BalanceSheet.objects
        .filter(user=request.user)
        .order_by('-added_at')
    )

    stats = {
        "total": sheets.count(),
        "open_items": sheets.filter(recon_status__iexact='open').count(),
        "flagged": sheets.exclude(flag_color__isnull=True).exclude(flag_color='').count(),
        "analysis_required": sheets.filter(analysis_required__iexact='yes').count(),
    }

    context = {
        "balance_sheets": sheets,
        "stats": stats,
    }
    return render(request, 'gl_reviews/balance_sheet.html', context)

