from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()
    noise_type = forms.ChoiceField(choices=[('gaussian', 'Gaussian'), ('salt_pepper', 'Salt and Pepper')])
    denoise_type = forms.ChoiceField(choices=[('mean', 'Mean'), ('median', 'Median')])
    sharp_type = forms.ChoiceField(choices=[('lap', 'Lalacian Filter'), ('high_pass', 'High pass filter'), ('kernel', 'Kernel-based sharpening')])
