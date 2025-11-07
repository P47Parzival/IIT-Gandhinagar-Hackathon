from django.contrib.auth.decorators import login_required
from django.shortcuts import render


@login_required
def reports_view(request):
    """Static reports landing page."""

    # Static data mirroring the RACI-style matrix shown in the reference image
    roles = [
        {"label": "Project Lead", "name": "Anne"},
        {"label": "Internal Recruiter", "name": "John"},
        {"label": "Hiring Manager", "name": "Natasha"},
        {"label": "Stakeholder 4", "name": "Steven"},
        {"label": "Stakeholder 5", "name": "Sarah"},
        {"label": "Stakeholder 8", "name": "Allison"},
    ]

    raw_tasks = [
        {
            "title": "Task 1",
            "caption": "Defining the job role",
            "assignments": ["A", "A", "R", "I", "R", "A"],
        },
        {
            "title": "Task 2",
            "caption": "Creating a requisition",
            "assignments": ["A", "R", "I", "C", "I", "A"],
        },
        {
            "title": "Task 3",
            "caption": "Writing the job ad",
            "assignments": ["C", "A", "C", "A", "C", "C"],
        },
        {
            "title": "Task 4",
            "caption": "Posting the job ad",
            "assignments": ["C", "R", "I", "R", "I", "C"],
        },
        {
            "title": "Task 5",
            "caption": "Promote the position on the company channels",
            "assignments": ["C", "A", "I", "R", "I", "C"],
        },
        {
            "title": "Task 6",
            "caption": "Advertise the position internally",
            "assignments": ["I", "A", "R", "C", "R", "I"],
        },
        {
            "title": "Task 7",
            "caption": "Review applications",
            "assignments": ["A", "I", "R", "I", "R", "A"],
        },
        {
            "title": "Task 8",
            "caption": "Candidate screening",
            "assignments": ["C", "I", "C", "I", "C", "C"],
        },
    ]

    matrix_rows = []
    for task in raw_tasks:
        cells = []
        for role, assignment in zip(roles, task["assignments"]):
            cells.append(
                {
                    "assignment": assignment,
                    "role_label": f"{role['label']} — {role['name']}",
                }
            )

        matrix_rows.append(
            {
                "title": task["title"],
                "caption": task["caption"],
                "cells": cells,
            }
        )

    context = {
        "roles": roles,
        "tasks": matrix_rows,
    }
    return render(request, 'reports/reports.html', context)

