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
def link_sap_erp_to_unified_db(request, saplink_id):
    """Connect to SAP HANA and fetch data."""
    user = request.user
    try:
        saplink = SAPLink.objects.get(id=saplink_id, link__user=user)
        
        if saplink.system_type == 'sap_hana':
            # Parse hostname (remove port if included)
            host_name = saplink.hana_host[:4]
            port = int(saplink.hana_port)
            
            print(f"Connecting to: {host_name}:{port}")
            
            # Connection parameters for SAP HANA Cloud
            connection_params = {
                'address': host_name,
                'port': port,
                'user': saplink.username,
                'password': saplink.password,
                'encrypt': True,
                'sslValidateCertificate': False,  # For development only
            }
            
            # For SAP HANA Cloud, add these parameters
            if 'hanacloud.ondemand.com' in host_name:
                connection_params.update({
                    'sslCryptoProvider': 'openssl',
                    'sslTrustStore': None,  # Use system trust store
                })
            
            # Connect to SAP HANA
            conn = dbapi.connect(**connection_params)
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute("SELECT GL_CODE, GL_NAME, AMOUNT FROM FINNOVATE_ERP.TRIAL_BALANCE ORDER BY GL_CODE LIMIT 5")
            rows = cursor.fetchall()
            
            # Log results
            print("=== SAMPLE TRIAL BALANCE RECORDS FROM HANA ===")
            for r in rows:
                print(r)
            
            cursor.close()
            conn.close()
            
            messages.success(request, f"Connected to SAP HANA successfully! {len(rows)} rows fetched.")
            return redirect('link_data_page')
            
    except SAPLink.DoesNotExist:
        messages.error(request, "SAP Link not found.")
        return redirect('link_data_page')
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print("SAP HANA connection error:", error_details)
        messages.error(request, f"Connection failed: {str(e)}")
        return redirect('link_data_page')