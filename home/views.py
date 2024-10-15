import base64
import cv2
import numpy as np
from django.shortcuts import render
from .forms import ImageUploadForm
from . import functions

def tobytes_img(image):
    _, buffer = cv2.imencode('.png', image)
    image_bytes = buffer.tobytes()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_base64
    

def add_noise(image, noise_type):
    if noise_type == 'gaussian':
        noisy_image = functions.gaussian_noise(image)
    elif noise_type == 'salt_pepper':
        noisy_image = functions.snp_noise(image)
    else:
        noisy_image = image
    return noisy_image

def denoise(image, denoise_type):
    if denoise_type == 'mean':
        denoise_image = functions.mean_blur(image)
    elif denoise_type == 'median':
        denoise_image = functions.median_blur(image)
    else:
        denoise_image = image
    return denoise_image

def sharp(image, sharp_type):
    if sharp_type == 'lap':
        sharp_image = functions.laplacian_sharp(image)
    elif sharp_type == 'high_pass':
        sharp_image = functions.high_pass_sharp(image)
    else:
        sharp_image = image
    return sharp_image

def extract_edge_feature(image):
    sobel = functions.sobel_features(image)
    prewitt = functions.prewitt_extract(image)
    canny = functions.canny_extract(image)
    return sobel, prewitt, canny


def upload_image(request):
    original_image_base64 = None
    noisy_image_base64 = None
    denoise_image_base64 = None
    sharp_image_base64 = None
    sobel_image_base64 = None
    prewitt_image_base64 = None
    canny_image_base64 = None
    
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            noise_type = form.cleaned_data['noise_type']
            denoise_type = form.cleaned_data['denoise_type']
            sharp_type = form.cleaned_data['sharp_type']
            
            # Đọc và giải mã ảnh gốc
            img_array = np.frombuffer(image.read(), np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Thêm noise vào ảnh
            noisy_image = add_noise(img, noise_type)
            
            # Denoise ảnh
            denoise_image = denoise(noisy_image, denoise_type)
            
            # Sharpening anhr
            sharp_image = sharp(denoise_image, sharp_type)
            
            # Extract Edge features
            sobel, prewitt, canny = extract_edge_feature(sharp_image)
            
            # Mã hóa ảnh gốc thành chuỗi byte
            original_image_base64 = tobytes_img(img)
            noisy_image_base64 = tobytes_img(noisy_image)
            denoise_image_base64 = tobytes_img(denoise_image)
            sharp_image_base64 = tobytes_img(sharp_image)
            sobel_image_base64 = tobytes_img(sobel)
            prewitt_image_base64 = tobytes_img(prewitt)
            canny_image_base64 = tobytes_img(canny)
               
            # Truyền ảnh vào template
            return render(request, 'index.html', {
                'form': form,
                'original_image': original_image_base64,
                'noisy_image': noisy_image_base64,
                'denoise_image': denoise_image_base64,
                'sharp_image': sharp_image_base64,
                'sobel_image': sobel_image_base64,
                'prewitt_image': prewitt_image_base64,
                'canny_image': canny_image_base64
            })
    else:
        form = ImageUploadForm()
        
    # Truyền ảnh vào template
    return render(request, 'index.html', {
        'form': form,
    })

