from django.urls import path
from .link_data import (
    link_data_view,
    handle_upload,
    link_data_connect_erp,
    link_data_connect_api,
    link_sap_erp_to_unified_db,
    get_sap_columns,
)

urlpatterns = [
    path('', link_data_view, name='link_data_page'),
    path('upload/', handle_upload, name='link_data_upload'),
    path('connect-erp/', link_data_connect_erp, name='link_data_connect_erp'),
    path('connect-api/', link_data_connect_api, name='link_data_connect_api'),

    path('link_sap_erp_to_unified_db/<uuid:saplink_id>/<str:table_name>/', link_sap_erp_to_unified_db, name='link_sap_erp_to_unified_db'),

    path('get_columns/<uuid:saplink_id>/<str:table_name>/', get_sap_columns, name='get_sap_columns'),
]
