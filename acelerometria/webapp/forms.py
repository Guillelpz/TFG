from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from .models import File
from django.contrib.auth.forms import SetPasswordForm, PasswordChangeForm
from django.utils.translation import gettext_lazy as _

class UploadFileForm(forms.ModelForm):
    class Meta:
        model = File
        fields = [
            'upload', 'age', 'weight', 'height', 'gender',
            'device_type', 'attachment_site', 'sampling_frequency'
        ]
        widgets = {
            'gender': forms.Select(choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')]),
            'attachment_site': forms.TextInput(attrs={'placeholder': 'e.g., Wrist, Hip'}),
        }

class RegistroUsuarioForm(UserCreationForm):
    email = forms.EmailField(required=True, help_text="Introduce a valid email.")

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email

class CustomSetPasswordForm(SetPasswordForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['new_password1'].label = "New password"
        self.fields['new_password1'].help_text = (
            "Your password must be at least 8 characters long, should not be too common, "
            "and should not be entirely numeric."
        )

        self.fields['new_password2'].label = "Confirm new password"
        self.fields['new_password2'].help_text = "Enter the same password again for verification."


    def clean_new_password2(self):
        password1 = self.cleaned_data.get("new_password1")
        password2 = self.cleaned_data.get("new_password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("The two passwords did not match.")
        return password2

class CustomPasswordChangeForm(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fields['old_password'].label = "Current password"
        self.fields['old_password'].help_text = "Enter your current password to authorize the change."

        self.fields['new_password1'].label = "New password"
        self.fields['new_password1'].help_text = (
            "Your password must be at least 8 characters long, should not be too common, "
            "and should not be entirely numeric."
        )

        self.fields['new_password2'].label = "Confirm new password"
        self.fields['new_password2'].help_text = "Enter the same password again for verification."

    def clean_new_password2(self):
        password1 = self.cleaned_data.get("new_password1")
        password2 = self.cleaned_data.get("new_password2")
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("The two passwords did not match.")
        return password2
