from django.db import models
from django.contrib.auth.models import User

class File(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    upload = models.FileField(upload_to='uploads/')
    created_at = models.DateTimeField(auto_now_add=True)

    # Metadata about the person who generated the data
    age = models.IntegerField(null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)

    # Metadata about the device and recording
    device_type = models.CharField(max_length=100, null=True, blank=True)
    attachment_site = models.CharField(max_length=100, null=True, blank=True)
    sampling_frequency = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.upload.name} ({self.user.username}) - {self.age} y/o, {self.gender}"
    
class Metric(models.Model):
    file = models.ForeignKey(File, on_delete=models.CASCADE, related_name='metrics')
    timestamp = models.DateTimeField()

    # Sensor metrics
    ENMO = models.FloatField(null=True, blank=True)
    anglex = models.FloatField(null=True, blank=True)
    angley = models.FloatField(null=True, blank=True)
    anglez = models.FloatField(null=True, blank=True)
    MAD = models.FloatField(null=True, blank=True)
    NeishabouriCount_x = models.FloatField(null=True, blank=True)
    NeishabouriCount_y = models.FloatField(null=True, blank=True)
    NeishabouriCount_z = models.FloatField(null=True, blank=True)
    NeishabouriCount_vm = models.FloatField(null=True, blank=True)

    # Activity label assigned manually (from graph tagging)
    activity = models.CharField(max_length=300, null=True, blank=True)

    def __str__(self):
        activity_info = f" - {self.activity}" if self.activity else ""
        return f"{self.timestamp} - {self.file.upload.name}{activity_info}"

