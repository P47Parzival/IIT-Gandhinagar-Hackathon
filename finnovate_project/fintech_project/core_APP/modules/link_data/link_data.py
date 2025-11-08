import logging
from hdbcli import dbapi
import os
from django.shortcuts import render
from django.contrib.auth import authenticate, login
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from core_APP.modules.link_data.link_data_forms import UploadedFileForm, SAPLinkForm
from core_APP.models import LinkedData, UploadedFile, SAPLink
from django.db import transaction
from django.db.models import OuterRef, Subquery
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import json
import traceback
from core_APP.models import TrialBalance, BalanceSheet


logger = logging.getLogger(__name__)


EXT_TO_SOURCE = {
    ".pdf": "pdf_file",
    ".xlsx": "xlsx_file",
    ".csv": "csv_file",
    ".xls": "xls_file",
    ".txt": "txt_file",
    ".docx": "docx_file",
    ".xlsm": "xlsm_file",
    ".xml": "xml_file",
    ".json": "json_file",
}


@login_required
def link_data_view(request):
    """Link Data view."""
    file_form = UploadedFileForm()
    sap_form = SAPLinkForm()

    # List of extensions that map to LinkedData sources
    ext_to_source_ls = list(EXT_TO_SOURCE.values())

    # --- FILES ---
    data_id_ls = []
    linked_data_qs = LinkedData.objects.filter(
        user=request.user,
        data_source__in=ext_to_source_ls
    )
    for ld in linked_data_qs:
        data_id_ls.append(ld.data_id)

    uploaded_files_qs = UploadedFile.objects.filter(
        id__in=data_id_ls,
        user=request.user
    )

    EXT_TO_READABLE = {
        ".pdf": "PDF File",
        ".xlsx": "Excel File (.xlsx)",
        ".csv": "CSV File",
        ".xls": "Excel File (.xls)",
        ".txt": "Text File",
        ".docx": "Word Document",
        ".xlsm": "Macro Excel File",
        ".xml": "XML File",
        ".json": "JSON File",
    }

    for f in uploaded_files_qs:
        ext = os.path.splitext(f.file.name)[1].lower()
        f.data_source = EXT_TO_READABLE.get(ext, "Unknown Type")

    # --- SAP LINKS ---
    sap_linked_data_qs = LinkedData.objects.filter(
        user=request.user,
        data_source='sap_erp'
    )
    sap_links_qs = SAPLink.objects.filter(link__in=sap_linked_data_qs)

    context = {
        "file_form": file_form,
        "sap_form": sap_form,
        "uploaded_files_qs": uploaded_files_qs,
        "sap_links_qs": sap_links_qs,
    }

    return render(request, "link_data/link_data.html", context)


@login_required
def handle_upload(request):
    if request.method == "POST":
        form = UploadedFileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded = request.FILES["file"]
            _, ext = os.path.splitext(uploaded.name)
            ext = ext.lower()

            data_source = EXT_TO_SOURCE.get(ext)
            if not data_source:
                return redirect("link_data_page")

            with transaction.atomic():
                uf = form.save(commit=False)
                uf.user = request.user
                uf.save()

                LinkedData.objects.create(
                    user=request.user,
                    data_source=data_source,
                    data_id=str(uf.id),
                )
            return redirect("link_data_page")
    return redirect("link_data_page")


@login_required
def link_data_connect_erp(request):
    """Handle SAP ERP connection submission."""
    if request.method == "POST":
        form = SAPLinkForm(request.POST)
        if form.is_valid():
            # create LinkedData entry
            linked_data = LinkedData.objects.create(
                user=request.user,
                data_source='sap_erp'
            )

            # create SAPLink
            sap_link = form.save(commit=False)
            sap_link.link = linked_data
            sap_link.save()

            # save LinkedData record with SAPLink id
            linked_data.data_id = str(sap_link.id)
            linked_data.save(update_fields=["data_id"])

            messages.success(request, f"SAP ERP system '{sap_link.system_name}' linked successfully.")
        else:
            messages.error(request, "Failed to link SAP system. Please correct the errors.")
    return redirect('link_data_page')


@login_required
def link_data_connect_api(request):
    """Connect API."""
    return redirect('link_data_page')


@login_required
@csrf_exempt
def link_sap_erp_to_unified_db(request, saplink_id, table_name):
    """
    Connect to SAP HANA, fetch column metadata (GET)
    or import mapped data to local DB (POST).
    """
    user = request.user
    try:
        print("Inside link_sap_erp_to_unified_db")
        print("saplink_id:", saplink_id)
        print("table_name:", table_name)
        # table_name can be trial_balance or balance_sheet
        saplink = SAPLink.objects.get(id=saplink_id, link__user=user)

        # Only for SAP HANA systems
        if saplink.system_type != 'sap_hana':
            return JsonResponse({"error": "Only SAP HANA connections are supported."}, status=400)

        print("saplink.hana_host:", saplink.hana_host)
        host_name = saplink.hana_host[:-4]
        port = int(saplink.hana_port)
        print(f"Connecting to: {host_name}:{port}")

        connection_params = {
            'address': host_name,
            'port': port,
            'user': saplink.username,
            'password': saplink.password,
            'encrypt': True,
            'sslValidateCertificate': False,  # For development only
        }

        if 'hanacloud.ondemand.com' in host_name:
            connection_params.update({
                'sslCryptoProvider': 'openssl',
                'sslTrustStore': None,  # Use system trust store
            })

        # Handle GET: fetch SAP column metadata
        if request.method == "GET":
            conn = dbapi.connect(**connection_params)
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME
                FROM SYS.TABLE_COLUMNS
                WHERE SCHEMA_NAME = '{saplink.hana_database}'
                AND TABLE_NAME = '{table_name.upper()}'
                ORDER BY POSITION
            """)
            columns = [{"name": r[0], "type": r[1]} for r in cursor.fetchall()]
            cursor.close()
            conn.close()

            # Define local Django fields for mapping
            if table_name == "trial_balance":
                local_fields = [
                    "gl_code", "gl_name", "group_gl_code", "group_gl_name",
                    "amount", "fs_main_head", "fs_sub_head", "fiscal_year",
                ]
            else:
                local_fields = [
                    "BS_PL","status","gl_acct","gl_account_name","main_head","sub_head",
                    "cml","frequency","responsible_department","department_spoc","department_reviewer",
                    "query_type_action_points","working_needed","confirmation_type","recon_status",
                    "variance_percent","flag_color","report_type","analysis_required","review_checkpoint_abex","fiscal_year"
                ]

            return JsonResponse({"columns": columns, "local_fields": local_fields})

        # Handle POST: import data using mapping
        elif request.method == "POST":
            body = json.loads(request.body.decode("utf-8"))
            mapping = body.get("mapping", {})

            conn = dbapi.connect(**connection_params)
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM "{saplink.hana_database}"."{table_name.upper()}"')
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            cursor.close()
            conn.close()

            inserted_count = 0

            with transaction.atomic():
                if table_name == "trial_balance":
                    objs = []
                    for row in rows:
                        row_dict = dict(zip(col_names, row))
                        objs.append(TrialBalance(
                            user=user,
                            gl_code=row_dict.get(mapping.get("gl_code")),
                            gl_name=row_dict.get(mapping.get("gl_name")),
                            group_gl_code=row_dict.get(mapping.get("group_gl_code")),
                            group_gl_name=row_dict.get(mapping.get("group_gl_name")),
                            amount=row_dict.get(mapping.get("amount")) or 0,
                            fs_main_head=row_dict.get(mapping.get("fs_main_head")),
                            fs_sub_head=row_dict.get(mapping.get("fs_sub_head")),
                            fiscal_year=row_dict.get(mapping.get("fiscal_year")),
                        ))
                    TrialBalance.objects.bulk_create(objs)
                    inserted_count = len(objs)

                elif table_name == "balance_sheet":
                    objs = []
                    for row in rows:
                        row_dict = dict(zip(col_names, row))
                        objs.append(BalanceSheet(
                            user=user,
                            BS_PL=row_dict.get(mapping.get("BS_PL")),
                            status=row_dict.get(mapping.get("status")),
                            gl_acct=row_dict.get(mapping.get("gl_acct")),
                            gl_account_name=row_dict.get(mapping.get("gl_account_name")),
                            main_head=row_dict.get(mapping.get("main_head")),
                            sub_head=row_dict.get(mapping.get("sub_head")),
                            cml=row_dict.get(mapping.get("cml")),
                            frequency=row_dict.get(mapping.get("frequency")),
                            responsible_department=row_dict.get(mapping.get("responsible_department")),
                            department_spoc=row_dict.get(mapping.get("department_spoc")),
                            department_reviewer=row_dict.get(mapping.get("department_reviewer")),
                            query_type_action_points=row_dict.get(mapping.get("query_type_action_points")),
                            working_needed=row_dict.get(mapping.get("working_needed")),
                            confirmation_type=row_dict.get(mapping.get("confirmation_type")),
                            recon_status=row_dict.get(mapping.get("recon_status")),
                            variance_percent=row_dict.get(mapping.get("variance_percent")),
                            flag_color=row_dict.get(mapping.get("flag_color")),
                            report_type=row_dict.get(mapping.get("report_type")),
                            analysis_required=row_dict.get(mapping.get("analysis_required")),
                            review_checkpoint_abex=row_dict.get(mapping.get("review_checkpoint_abex")),
                            fiscal_year=row_dict.get(mapping.get("fiscal_year")),
                        ))
                    BalanceSheet.objects.bulk_create(objs)
                    inserted_count = len(objs)

                # Update SAPLink status to "imported"
                saplink.status[table_name] = "imported"
                saplink.save(update_fields=["status"])

            return JsonResponse({"success": True, "imported_count": inserted_count})

        else:
            return JsonResponse({"error": "Invalid request method."}, status=405)

    except SAPLink.DoesNotExist:
        return JsonResponse({"error": "SAP Link not found."}, status=404)

    except Exception as e:
        print("SAP HANA connection error:", traceback.format_exc())
        return JsonResponse({"error": str(e)}, status=500)
    

@login_required
def get_sap_columns(request, saplink_id, table_name):
    """AJAX endpoint: fetch SAP table columns."""
    try:
        saplink = SAPLink.objects.get(id=saplink_id, link__user=request.user)
        conn = dbapi.connect(
            address=saplink.hana_host[:-4],
            port=int(saplink.hana_port),
            user=saplink.username,
            password=saplink.password,
            encrypt=True,
            sslValidateCertificate=False,
        )
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT TABLE_NAME, SCHEMA_NAME
            FROM SYS.TABLES
            WHERE SCHEMA_NAME = '{saplink.hana_database}'
        """)
        print([r[0] for r in cursor.fetchall()])
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE_NAME
            FROM SYS.TABLE_COLUMNS
            WHERE SCHEMA_NAME = '{saplink.hana_database}'
            AND TABLE_NAME = UPPER('{table_name}')
            ORDER BY POSITION
        """)
        columns = [{"name": r[0], "type": r[1]} for r in cursor.fetchall()]
        columns = [d for d in columns if d.get('name') != 'ID']

        print(columns)
        cursor.close()
        conn.close()

        if table_name == "trial_balance":
            local_fields = [
                "gl_code", "gl_name", "group_gl_code", "group_gl_name",
                "amount", "fs_main_head", "fs_sub_head", "fiscal_year",
            ]
        else:
            local_fields = [
                "BS_PL","status","gl_acct","gl_account_name","main_head","sub_head",
                "cml","frequency","responsible_department","department_spoc","department_reviewer",
                "query_type_action_points","working_needed","confirmation_type","recon_status",
                "variance_percent","flag_color","report_type","analysis_required","review_checkpoint_abex","fiscal_year"
            ]

        return JsonResponse({"columns": columns, "local_fields": local_fields})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)