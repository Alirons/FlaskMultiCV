from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix
from skimage.feature import hog
from skimage import exposure
from skimage import io

import pickle
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torchvision import transforms

import timm
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

class ModelCONFIG:
    size_w = 224  # 图像的宽度（像素）
    size_h = 224  # 图像的高度（像素）
    model_name = 'best_score.pth'  # 模型文件名
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 确保上传和静态文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

'''
ResNet50
'''
class CustomModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # 创建一个ResNet-50模型
        self.model = timm.create_model('resnet50', pretrained=pretrained, in_chans=3)
        self.n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Conv2d(self.n_features, 44, 1))

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pool_feature = self.pooling(features)
        output = self.classifier(pool_feature).view(bs, -1)
        return output

def load_model(model_name, device):

    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)
    checkpoint = torch.load(model_path, map_location=device)

    model = CustomModel()

    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model

# 检查文件扩展名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# 上传文件的路由
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('display_image', filename=filename))
    return render_template('index.html')

# 显示图片的路由
@app.route('/uploads/<filename>')
def display_image(filename):
    file_url = url_for('uploaded_file', filename=filename)
    return render_template('display.html', file_url=file_url, filename=filename)

# 处理图片和生成直方图的路由
@app.route('/analyze/<filename>')
def analyze_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    rgb_name, hsv_name, lab_name, lbp_name, edi_name, glcm_name, sift_name, hog_name = process_the_image(file_path)
    return render_template('resultA.html', 
                           rgb_image=rgb_name, 
                           hsv_image=hsv_name, 
                           lab_image=lab_name, 
                           lbp_image=lbp_name, 
                           edi_image=edi_name, 
                           glcm_image=glcm_name, 
                           sift_image=sift_name, 
                           hog_image=hog_name, 
                           filename=filename)

@app.route('/cluster/<filename>', methods=['GET', 'POST'])
def cluster_pixels(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    lpdf, kmeans_name, spr_kmeans_name = segment_the_image(file_path)
    lpdf_html = lpdf.to_html(classes='data', header="true", index=False)
    return render_template('resultB.html', filename=filename, kmeans_image=kmeans_name, spr_kmeans_image=spr_kmeans_name, table=lpdf_html)

@app.route('/classify/<filename>', methods=['GET', 'POST'])
def classify_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predicted_class = predict_the_image(file_path, ModelCONFIG.device)
    categories = ['Category 0', 'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6', 'Category 7', 'Category 8', 'Category 9', 'Category 10']
    predicted_category = categories[predicted_class]
    return render_template('resultC.html', filename=filename, prediction=predicted_category)


# 提供上传文件的服务
@app.route('/uploads/files/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_the_image(image_path):

    pilimage = Image.open(image_path)

    iograyimage = io.imread(image_path, as_gray=True)
    iograyimage255 = (iograyimage * 255).astype('uint8')

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2Lab)
    
    rgb_fullname = generate_histogram_rgb(pilimage, image_path)
    hsv_fullname = generate_histogram_hsv(hsv_image, image_path)
    lab_fullname = generate_histogram_lab(lab_image, image_path)

    lbp_fullname = generate_histogram_lbp(gray_image, 8, 1, 'uniform', image_path)
    edi_fullname = generate_edge_detection_image(gray_image, image_path)

    glcm_fullname = visualize_glcm(iograyimage255, image_path)

    sift_fullname = visualize_sift(gray_image, image_path)

    hog_fullname = visualize_hog(iograyimage, image_path)

    return rgb_fullname, hsv_fullname, lab_fullname, lbp_fullname, edi_fullname, glcm_fullname, sift_fullname, hog_fullname


def segment_the_image(image_path):

    ioimage = io.imread(image_path)

    lp_statistics, kmeans_fullname = kmeans_clustering_segmentation('kmeans_model.pkl', image_path)
    separated_clusters_name = visualize_clusters_separately('kmeans_model.pkl', ioimage, image_path)

    return lp_statistics, kmeans_fullname, separated_clusters_name


def predict_the_image(image_path, device):

    transform = transforms.Compose([
        transforms.Resize((ModelCONFIG.size_w, ModelCONFIG.size_h)),
        transforms.ToTensor(),
    ])

    pilimage = Image.open(image_path)
    image = transform(pilimage)
    image = image.unsqueeze(0).to(device)

    model = load_model(ModelCONFIG.model_name, device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

'''
RGB
'''
def generate_histogram_rgb(image, image_path):

    r, g, b = image.split()

    r_histogram = r.histogram()
    g_histogram = g.histogram()
    b_histogram = b.histogram()

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.7, wspace=0.4, hspace=0.1)

    axs[0, 0].imshow(r, cmap='Reds')
    axs[0, 0].set_title('Red Channel')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(g, cmap='Greens')
    axs[0, 1].set_title('Green Channel')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(b, cmap='Blues')
    axs[0, 2].set_title('Blue Channel')
    axs[0, 2].axis('off')

    axs[1, 0].bar(range(256), r_histogram, color='red', alpha=0.6, width=1.0)
    axs[1, 0].set_title('Red Channel Histogram')
    axs[1, 0].set_xlim([0, 256])

    axs[1, 1].bar(range(256), g_histogram, color='green', alpha=0.6, width=1.0)
    axs[1, 1].set_title('Green Channel Histogram')
    axs[1, 1].set_xlim([0, 256])

    axs[1, 2].bar(range(256), b_histogram, color='blue', alpha=0.6, width=1.0)
    axs[1, 2].set_title('Blue Channel Histogram')
    axs[1, 2].set_xlim([0, 256])

    hist_filename = f'RGB_hist_{os.path.basename(image_path)}.png'
    hist_path = os.path.join('static', hist_filename)
    fig.savefig(hist_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return hist_filename

'''
HSV
'''
def generate_histogram_hsv(image, image_path):
    
    H, S, V = cv2.split(image)

    hist_hue = cv2.calcHist([image], [0], None, [180], [0, 180])
    hist_saturation = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([image], [2], None, [256], [0, 256])

    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.7, wspace=0.4, hspace=0.1)
    
    plt.subplot(231)
    plt.imshow(H, cmap='hsv')
    plt.title('Hue Channel')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(S, cmap='hot')
    plt.title('Saturation Channel')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(V, cmap='hot')
    plt.title('Value Channel')
    plt.axis('off')

    plt.subplot(234)
    plt.title('Hue Histogram')
    plt.bar(range(180), hist_hue.flatten(), width=1)
    plt.xlim([0, 180])

    plt.subplot(235)
    plt.title('Saturation Histogram')
    plt.bar(range(256), hist_saturation.flatten(), width=1)
    plt.xlim([0, 256])

    plt.subplot(236)
    plt.title('Value Histogram')
    plt.bar(range(256), hist_value.flatten(), width=1)
    plt.xlim([0, 256])

    hist_filename = f'HSV_hist_{os.path.basename(image_path)}.png'
    hist_path = os.path.join('static', hist_filename)
    plt.savefig(hist_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

    return hist_filename

'''
Lab
'''
def generate_histogram_lab(image, image_path):

    L, a, b = cv2.split(image)

    L_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    a_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    b_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.7, wspace=0.4, hspace=0.1)

    plt.subplot(231)
    plt.imshow(L, cmap='gray')
    plt.title('L Channel')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(a, cmap='hot')
    plt.title('a Channel')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(b, cmap='hot')
    plt.title('b Channel')
    plt.axis('off')

    plt.subplot(234)
    plt.title('L Histogram')
    plt.bar(range(256), L_hist.flatten(), width=1)
    plt.xlim([0, 256])

    plt.subplot(235)
    plt.title('a Histogram')
    plt.bar(range(256), a_hist.flatten(), width=1)
    plt.xlim([0, 256])

    plt.subplot(236)
    plt.title('b Histogram')
    plt.bar(range(256), b_hist.flatten(), width=1)
    plt.xlim([0, 256])

    hist_filename = f'Lab_hist_{os.path.basename(image_path)}.png'
    hist_path = os.path.join('static', hist_filename)
    plt.savefig(hist_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

    return hist_filename
'''
LBP
'''
def generate_histogram_lbp(image, n_points, radius, METHOD, image_path):

    lbp = local_binary_pattern(image, n_points*3, radius*3, METHOD)

    n_bins = int(lbp.max() + 1)
    
    hist, _ = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(0, n_bins), hist, width=1.0, color='skyblue', edgecolor='black')

    plt.title("LBP Histogram")
    plt.xlabel("LBP Value")
    plt.ylabel("Percentage")

    hist_filename = f'LBP_hist_{os.path.basename(image_path)}.png'
    hist_path = os.path.join('static', hist_filename)
    plt.savefig(hist_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

    return hist_filename

'''
EDI
'''
def generate_edge_detection_image(image, image_path):
    
    edges = cv2.Canny(image, 100, 200)

    edge_pixels = np.sum(edges > 0)
    total_pixels = image.size
    edge_density = edge_pixels / total_pixels

    fig, ax = plt.subplots()
    
    ax.imshow(edges, cmap='gray')
    ax.set_title('Edge Detection Image')
    ax.axis('off')

    text = f'Edge Density: {edge_density:.4f}'
    fig.text(0.5, -0.05, text, ha='center', fontsize=12, color='red', fontweight='bold', transform=ax.transAxes)
    
    fig.subplots_adjust(bottom=0.2)

    map_filename = f'edi_map_{os.path.basename(image_path)}.png'
    map_path = os.path.join('static', map_filename)
    fig.savefig(map_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return map_filename

'''
GLCM
'''
def visualize_glcm(image, image_path):
    
    distances = [1]  # 距离参数
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 角度参数
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    fig, axes = plt.subplots(1, len(angles), figsize=(10, 5))
    for i, angle in enumerate(angles):
        ax = axes[i]
        im = ax.imshow(glcm[:, :, 0, i], cmap='hot')
        ax.set_title(f'Angle: {np.degrees(angle):.0f} degrees')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    fig.tight_layout()

    matrix_filename = f'glcm_{os.path.basename(image_path)}.png'
    matrix_path = os.path.join('static', matrix_filename)
    fig.savefig(matrix_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return matrix_filename

'''
SIFT
'''
def visualize_sift(image, image_path):

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(image, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    sift_image = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(2, 2))
    plt.title("SIFT Features", fontsize=6)
    plt.imshow(sift_image)
    plt.axis('off')

    img_filename = f'sift_{os.path.basename(image_path)}.png'
    img_path = os.path.join('static', img_filename)
    plt.savefig(img_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

    return img_filename

'''
HOG
'''
def visualize_hog(image, image_path):

    hog_features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Original Image')

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    fig.tight_layout()

    img_filename = f'hog_{os.path.basename(image_path)}.png'
    img_path = os.path.join('static', img_filename)
    fig.savefig(img_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return img_filename

'''
HSVFeaturesbyPIL
'''
def extract_hsv_features(image_path):

    hsvimage = Image.open(image_path).convert("HSV")
    hsvarray = np.array(hsvimage)

    hsv_features = hsvarray.reshape(-1, 3).astype(float) / 255.0

    return hsv_features

'''
K-Means
'''
def kmeans_clustering_segmentation(model_name, image_path):

    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)

    with open(model_path, 'rb') as file:
        kmeans = pickle.load(file)

    features = extract_hsv_features(image_path)

    labels = kmeans.predict(features)
    clustered_image = labels.reshape(ModelCONFIG.size_w, ModelCONFIG.size_h)

    unique, counts = np.unique(clustered_image, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    dataframe = pd.DataFrame(list(cluster_counts.items()), columns=['Label', 'Pixel Count'])

    plt.imshow(clustered_image, cmap='coolwarm')
    plt.colorbar()
    plt.axis('off')

    img_filename = f'kmeans_clustering_{os.path.basename(image_path)}.png'
    img_path = os.path.join('static', img_filename)
    plt.savefig(img_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

    return dataframe, img_filename

'''
Separating K-Means Clustering Results
'''
def visualize_clusters_separately(model_name, image, image_path):
    model_path = os.path.join(app.config['MODEL_FOLDER'], model_name)

    with open(model_path, 'rb') as file:
        kmeans = pickle.load(file)

    features = extract_hsv_features(image_path)

    labels = kmeans.predict(features)
    clustered_image = labels.reshape(ModelCONFIG.size_w, ModelCONFIG.size_h)

    unique_labels = np.unique(clustered_image)

    num_labels = len(unique_labels)
    cols = 3
    rows = (num_labels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))

    for ax, label in zip(axes.flat, unique_labels):
        mask = clustered_image == label
        cluster_image = np.ones_like(image) * 255  # 用白色填充
        cluster_image[mask] = image[mask]
        
        ax.imshow(cluster_image)
        ax.axis('off')
        ax.set_title(f'Cluster {label}')

        # 添加黑色边框
        bbox = FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes, 
                              linewidth=1, edgecolor='black', facecolor='none', boxstyle='square,pad=0')
        ax.add_patch(bbox)

    for ax in axes.flat[num_labels:]:
        ax.remove()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01, top=0.80, bottom=0.05)
    separated_clusters_filename = f'separated_clusters_{os.path.basename(image_path)}.png'
    separated_clusters_path = os.path.join('static', separated_clusters_filename)
    plt.savefig(separated_clusters_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()

    return separated_clusters_filename

if __name__ == '__main__':
    app.run(debug=True)