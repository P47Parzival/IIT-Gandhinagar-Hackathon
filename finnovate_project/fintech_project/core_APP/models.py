from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
import uuid

# Create your models here.

class CustomUser(AbstractUser):
    USER_TYPE_CHOICES = (
        (1, 'Admin'),
        (2, 'Department Head'), # per dept head
        (3, 'BUFC'),
        (4, 'User')
    )
    user_type = models.PositiveSmallIntegerField(choices=USER_TYPE_CHOICES, default=2)

    def __str__(self):
        return self.username
    
    class Meta:
        db_table = 'Users'


class Department(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'departments'
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class ResponsibilityMatrix(models.Model):

    USER_ROLE_CHOICES = (
        (2, 'Department Head'),
        (3, 'BUFC'),
        (4, 'Preparer'),
        (5, 'Reviewer')
    )
    GL_CODE_STATUS_CHOICES = (
        (1, 'Pending'),
        (2, 'Done'),
        (3, 'Approved'),
        (4, 'Rejected')
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name="responsibility_matrix")
    department = models.ForeignKey(Department, on_delete=models.SET_NULL, null=True, blank=True)
    user_role = models.PositiveSmallIntegerField(choices=USER_ROLE_CHOICES, default=5)

    gl_code = models.CharField(max_length=50, null=True, blank=True)
    gl_code_status = models.PositiveSmallIntegerField(choices=GL_CODE_STATUS_CHOICES, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "responsibility_matrix"
        unique_together = ("user", "gl_code")
        ordering = ["user", "gl_code"]

    def __str__(self):
        return f"{self.user.username} - {self.gl_code}"


class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'conversations'
        ordering = ['-updated_at']


class Message(models.Model):
    ROLE_CHOICES = (
        ('user', 'user'),
        ('assistant', 'assistant'),
        ('system', 'system'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'messages'
        ordering = ['created_at']


class LinkedData(models.Model):
    data_source_options = (
        ('pdf_file', 'PDF File'),
        ('xlsx_file', 'XLSX File'),
        ('csv_file', 'CSV File'),
        ('google_sheet', 'Google Sheet'),
        ('xls_file', 'XLS File'),
        ('txt_file', 'Text File'),
        ('docx_file', 'DOCX File'),
        ('xlsm_file', 'XLSM File'),
        ('xml_file', 'XML File'),
        ('json_file', 'JSON File'),
        ('sap_erp', 'SAP ERP System'),
    )
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='linked_data')
    data_source = models.CharField(max_length=50, choices=data_source_options)
    data_id = models.CharField(max_length=255, null=True, blank=True)
    linked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'linked_data'
        ordering = ['-linked_at']


class UploadedFile(models.Model):
    TABLE_TYPE_CHOICES = [
        ("trial_balance", "Trial Balance"),
        ("balance_sheet", "Balance Sheet"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='uploaded_files')
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='pending')

    table_type = models.CharField(
        max_length=50,
        choices=TABLE_TYPE_CHOICES,
    )

    class Meta:
        db_table = 'uploaded_files'
        ordering = ['-uploaded_at']


class SAPLink(models.Model):
    AUTH_METHOD_CHOICES = [
        ('basic', 'Basic Authentication'),
        ('oauth2', 'OAuth 2.0'),
        ('saml', 'SAML SSO'),
    ]

    SYSTEM_TYPE_CHOICES = [
        ('sap_odata', 'SAP OData API'),
        ('sap_rfc', 'SAP RFC/BAPI'),
        ('sap_hana', 'SAP HANA DB'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    link = models.ForeignKey(LinkedData, on_delete=models.CASCADE, related_name='sap_links')
    system_type = models.CharField(max_length=20, choices=SYSTEM_TYPE_CHOICES, default='sap_odata')

    # Common
    system_name = models.CharField(max_length=100)
    base_url = models.URLField(blank=True, null=True, help_text="Required for OData only")
    client_id = models.CharField(max_length=10, null=True, blank=True, help_text="SAP Client ID (e.g., 100)")
    auth_method = models.CharField(max_length=10, choices=AUTH_METHOD_CHOICES, default='basic')
    username = models.CharField(max_length=255, blank=True, null=True)
    password = models.CharField(max_length=255, blank=True, null=True)
    oauth_token = models.TextField(blank=True, null=True)

    # RFC-specific
    ashost = models.CharField(max_length=100, null=True, blank=True, help_text="SAP Application Server Host")
    sysnr = models.CharField(max_length=2, null=True, blank=True, help_text="SAP System Number (e.g., 00)")
    language = models.CharField(max_length=2, null=True, blank=True, default='EN', help_text="Language key (e.g., EN)")

    # HANA-specific
    hana_host = models.CharField(max_length=100, null=True, blank=True, help_text="SAP HANA Hostname")
    hana_port = models.CharField(max_length=10, null=True, blank=True, help_text="SAP HANA Port Number")
    hana_database = models.CharField(max_length=50, null=True, blank=True, help_text="SAP HANA Database Name")
    ssl_mode = models.CharField(
        max_length=20,
        choices=[('enabled', 'Enabled'), ('disabled', 'Disabled')],
        default='enabled'
    )

    connected_at = models.DateTimeField(auto_now_add=True)
    last_synced_at = models.DateTimeField(blank=True, null=True)
    status = models.JSONField(
        default=dict,
        blank=True,
        null=True
    )

    class Meta:
        db_table = 'sap_erp_links'
        ordering = ['-connected_at']

    def __str__(self):
        return f"{self.system_name} ({self.system_type})"
    
    def save(self, *args, **kwargs):
        if not self.status or not isinstance(self.status, dict):
            self.status = {
                "trial_balance": "pending",
                "balance_sheet": "pending"
            }
        super().save(*args, **kwargs)


# TIME TO BUILD UNIFIED DBs, MOTHERFUC--

# CustomUser (Dept Head)
#    │
#    ├── TrialBalance (GL-level financials)
#    │      ├── GLSupportingDocument
#    │      ├── GLReview
#    │      └── ValidationLog
#    │
#    └── ResponsibilityMatrix (who reviews what)


# https://www.appsruntheworld.com/customers-database/purchases/view/adani-group-india-selects-sap-s-4-hana-for-erp-financial

#                       ┌──────────────────────┐
#                       │     SAP S/4HANA      │
#                       │ (Adani Internal ERP) │
#                       └──────────┬───────────┘
#                                  │
#                   Secure OData / API Integration
#                                  │
#               ┌──────────────────┴───────────────────┐
#               │                                      │
#    ┌─────────────────────┐              ┌────────────────────────────┐
#    │ Finnovate Backend   │              │  Mock SAP Server:-         │
#    │ - SAPLink table     │              │  - OData-style endpoints   │
#    │ - Data ingestion    │              │  - Dummy Adani-style data  │
#    │ - TrialBalance DB   │              └────────────────────────────┘
#    │ - AI Validation     │
#    │ - Dashboards & APIs │
#    └──────────┬──────────┘
#               │
#            Frontend


class TrialBalance(models.Model):
    """Simplified, realistic Trial Balance model aligned with your SAP HANA schema."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="trial_balances"
    )

    # --- GL Information ---
    gl_code = models.CharField(max_length=50)
    gl_name = models.CharField(max_length=255, null=True, blank=True)
    group_gl_code = models.CharField(max_length=50, null=True, blank=True)
    group_gl_name = models.CharField(max_length=255, null=True, blank=True)

    # --- Financial Data ---
    amount = models.DecimalField(max_digits=18, decimal_places=2, default=0)

    # --- Financial Statement Grouping ---
    fs_main_head = models.CharField(max_length=255, null=True, blank=True)
    fs_sub_head = models.CharField(max_length=255, null=True, blank=True)

    # --- YYYY ---
    fiscal_year = models.CharField(max_length=10, null=True, blank=True)

    # --- Audit ---
    added_at = models.DateTimeField(auto_now_add=True)

    # --- Supporting Docx ---
    supporting_document = models.FileField(upload_to="supporting_documents/", null=True, blank=True)

    class Meta:
        db_table = "trial_balances"
        ordering = ["-added_at"]

    def __str__(self):
        return f"{self.gl_code} - {self.gl_name} ({self.fs_main_head or 'Uncategorized'})"
    

class BalanceSheet(models.Model):
    """Simplified, realistic Balance Sheet model aligned with your SAP HANA schema."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="balance_sheets"
    )    
    BS_PL = models.CharField(max_length=10, null=True, blank=True)
    status = models.CharField(max_length=50, null=True, blank=True)
    gl_acct = models.CharField(max_length=20)
    gl_account_name = models.CharField(max_length=100, null=True, blank=True)
    main_head = models.CharField(max_length=100, null=True, blank=True)
    sub_head = models.CharField(max_length=100, null=True, blank=True)
    cml = models.CharField(max_length=20, null=True, blank=True)
    frequency = models.CharField(max_length=20, null=True, blank=True)
    responsible_department = models.CharField(max_length=50, null=True, blank=True)
    department_spoc = models.CharField(max_length=100, null=True, blank=True)
    department_reviewer = models.CharField(max_length=100, null=True, blank=True)
    query_type_action_points = models.TextField(blank=True, null=True)
    working_needed = models.TextField(blank=True, null=True)
    confirmation_type = models.CharField(max_length=100, null=True, blank=True)
    recon_status = models.CharField(max_length=100, null=True, blank=True)
    variance_percent = models.CharField(max_length=50, null=True, blank=True)
    flag_color = models.CharField(max_length=20, null=True, blank=True)
    report_type = models.CharField(max_length=100, null=True, blank=True)
    analysis_required = models.CharField(max_length=10, null=True, blank=True)
    review_checkpoint_abex = models.CharField(max_length=500, blank=True, null=True)
    
    # --- YYYY ---
    fiscal_year = models.CharField(max_length=10, null=True, blank=True)

    # --- Audit ---
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'balance_sheet'

    def __str__(self):
        return f"{self.BS_PL} - {self.gl_acct} ({self.status})"

class GLReview(models.Model):
    GL_CODE_STATUS_CHOICES = (
        (1, 'Pending'),
        (2, 'Awaiting Approval'),
        (3, 'Approved'),
        (4, 'Rejected')
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    trial_balance = models.ForeignKey(TrialBalance, on_delete=models.CASCADE, related_name="reviews")
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    status = models.PositiveSmallIntegerField(choices=GL_CODE_STATUS_CHOICES, default=1)
    reconciliation_notes = models.TextField(blank=True, null=True)
    reviewed_at = models.DateTimeField(auto_now_add=True, null=True, blank=True)

    class Meta:
        db_table = "gl_reviews"
        ordering = ["-reviewed_at"]

    def __str__(self):
        return f"{self.trial_balance.gl_code} - {self.status}"
    

class GLSupportingDocument(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    gl_review = models.ForeignKey(GLReview, on_delete=models.CASCADE, related_name="supporting_documents", null=True, blank=True)
    file = models.FileField(upload_to="gl_supporting/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "gl_supporting_documents"
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"Support for {self.trial_balance.gl_code}"


class ReviewTrail(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="review_trails"
    )
    reviewer_responsibility_matrix = models.ForeignKey(
        "ResponsibilityMatrix",
        on_delete=models.SET_NULL,
        null=True,
        related_name="review_trails"
    )
    gl_review = models.ForeignKey(
        "GLReview",
        on_delete=models.SET_NULL,
        null=True,
        related_name="review_trails"
    )
    previous_trail = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="next_trails"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]