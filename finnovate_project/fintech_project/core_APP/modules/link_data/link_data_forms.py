from django import forms
from core_APP.models import UploadedFile, SAPLink


class UploadedFileForm(forms.ModelForm):
    TABLE_TYPE_CHOICES = [
        ("trial_balance", "Trial Balance"),
        ("balance_sheet", "Balance Sheet"),
    ]

    table_type = forms.ChoiceField(
        choices=TABLE_TYPE_CHOICES,
        required=True,
        widget=forms.Select(attrs={'class': 'form-select'})
    )

    class Meta:
        model = UploadedFile
        fields = ['file', 'table_type']


class SAPLinkForm(forms.ModelForm):
    class Meta:
        model = SAPLink
        fields = [
            'system_type',
            'system_name',
            'base_url',
            'client_id',
            'auth_method',
            'username',
            'password',
            'ashost',
            'sysnr',
            'language',
            'hana_host',
            'hana_port',
            'hana_database',
            'ssl_mode',
        ]
        widgets = {
            'password': forms.PasswordInput(render_value=True),
            'system_type': forms.Select(attrs={'class': 'form-select', 'id': 'id_system_type'}),
            'auth_method': forms.Select(attrs={'class': 'form-select'}),
            'base_url': forms.URLInput(attrs={'placeholder': 'https://sap.company.com/odata/'}),
        }

    def clean(self):
        cleaned = super().clean()
        system_type = cleaned.get('system_type')

        # --- Validation based on type ---
        if system_type == 'sap_odata':
            if not cleaned.get('base_url'):
                raise forms.ValidationError("Base URL is required for SAP OData connections.")

        elif system_type == 'sap_rfc':
            if not cleaned.get('ashost') or not cleaned.get('sysnr'):
                raise forms.ValidationError("Application Server Host and System Number are required for SAP RFC connections.")

        elif system_type == 'sap_hana':
            if not cleaned.get('hana_host') or not cleaned.get('hana_port') or not cleaned.get('hana_database'):
                raise forms.ValidationError("Host, Port, and Database are required for SAP HANA connections.")

        return cleaned
