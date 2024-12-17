from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.db import IntegrityError
from .forms import UploadECGForm
from .models import ECGData, Diagnosis
from .analysis import analyze_ecg  # Import analyze_ecg function

@login_required
def upload_ecg(request):
    if request.method == 'POST':
        form = UploadECGForm(request.POST, request.FILES)
        if form.is_valid():
            ecg_data = form.save(commit=False)
            ecg_data.user = request.user
            ecg_data.save()

            # Call analysis function for each file
            probable_diagnosis1, confidence_level1 = analyze_ecg(ecg_data.ecg_file1)
            probable_diagnosis2, confidence_level2 = analyze_ecg(ecg_data.ecg_file2)

            try:
                # Save diagnosis results
                diagnosis1 = Diagnosis.objects.create(
                    ecg_data=ecg_data,
                    probable_diagnosis=probable_diagnosis1,
                    confidence_level=confidence_level1
                )

                diagnosis2 = Diagnosis.objects.create(
                    ecg_data=ecg_data,
                    probable_diagnosis=probable_diagnosis2,
                    confidence_level=confidence_level2
                )
            except IntegrityError:
                # Rollback if diagnosis creation fails
                ecg_data.delete()
                raise

            return redirect('diagnosis', ecg_id=ecg_data.id)
    else:
        form = UploadECGForm()
    return render(request, 'upload_ecg.html', {'form': form})



@login_required
def diagnosis(request, ecg_id):
    ecg_data = ECGData.objects.get(pk=ecg_id)
    diagnoses = ecg_data.diagnosis_set.all()
    return render(request, 'diagnosis.html', {'diagnoses': diagnoses})
