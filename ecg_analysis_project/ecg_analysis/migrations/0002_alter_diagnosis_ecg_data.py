# Generated by Django 5.0.6 on 2024-05-15 11:00

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ecg_analysis', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='diagnosis',
            name='ecg_data',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='diagnosis', to='ecg_analysis.ecgdata'),
        ),
    ]
