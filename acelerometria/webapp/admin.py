from django.contrib import admin

# Register your models here.
from .models import File, Metric

admin.site.register(File)
admin.site.register(Metric)
