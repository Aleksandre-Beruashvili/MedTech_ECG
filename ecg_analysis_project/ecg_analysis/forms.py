from django import forms
from .models import ECGData

class UploadECGForm(forms.ModelForm):
    class Meta:
        model = ECGData
        fields = ['ecg_file1', 'ecg_file2']
