# ecg_analysis/models.py
from django.db import models
from django.contrib.auth.models import User

class ECGData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    upload_date = models.DateTimeField(auto_now_add=True)
    ecg_file1 = models.FileField(upload_to='ecg_files/')
    ecg_file2 = models.FileField(upload_to='ecg_files/', blank=True, null=True)

class Diagnosis(models.Model):
    ecg_data = models.OneToOneField(ECGData, on_delete=models.CASCADE, related_name='diagnosis')
    probable_diagnosis = models.CharField(max_length=100)
    confidence_level = models.FloatField()
