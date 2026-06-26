import cv2
import numpy as np
from scipy.ndimage import convolve

TARGET_SIZE = 2048

# ==================== 辅助函数 ====================
def ternaryOp(condition, trueVal, falseVal):
    return trueVal if condition else falseVal

# ==================== 1. 双边滤波 ====================
def bilateral_filter(I, sigma_s, sigma_r):
    """对应 MATLAB 的 bilateral_filter"""
    # 转换为 [0,255] 范围进行双边滤波
    I_uint8 = (I * 255).astype(np.uint8)
    # OpenCV 的 bilateralFilter: (src, d, sigmaColor, sigmaSpace)
    # 注意：OpenCV 的 sigma 含义与 MATLAB 不同，需要调整
    d = int(2 * sigma_s * 2 + 1)
    if d < 3:
        d = 3
    J = cv2.bilateralFilter(I_uint8, d, sigma_r * 255, sigma_s * 2)
    return J.astype(np.float32) / 255.0

# ==================== 2. 背景优化模块 ====================
def background_optimization_core(I, sigma=2):
    """对应 MATLAB 的 background_optimization_core"""
    # 转换为 double [0,1]
    I_double = I.astype(np.float32) / 255.0
    
    # RGB → HSV
    HSV = cv2.cvtColor(I_double, cv2.COLOR_BGR2HSV)
    S = HSV[:, :, 1]
    
    # 自适应阈值分割
    S_uint8 = (S * 255).astype(np.uint8)
    _, BW = cv2.threshold(S_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    BW = BW / 255.0
    
    # 形态学开运算去除小噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    BW_uint8 = (BW * 255).astype(np.uint8)
    BW_uint8 = cv2.morphologyEx(BW_uint8, cv2.MORPH_OPEN, kernel)
    BW = BW_uint8 / 255.0
    
    # 边缘保持滤波（双边滤波）
    info_img = np.zeros_like(I_double)
    for c in range(3):
        channel = I_double[:, :, c]
        channel_smooth = bilateral_filter(channel, sigma, sigma * 0.1)
        # 只对背景区域 (BW==0) 应用平滑
        channel[BW == 0] = channel_smooth[BW == 0]
        info_img[:, :, c] = channel
    
    # 归一化到 [0,1]
    info_img = np.clip(info_img, 0, 1)
    return info_img

# ==================== 3. 图像类型检测 ====================
def detect_discrete_image(I):
    """对应 MATLAB 的 detect_discrete_image"""
    if len(I.shape) == 3:
        gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    else:
        gray = I
    
    gray = gray.astype(np.float32) / 255.0
    
    # 自适应阈值
    gray_uint8 = (gray * 255).astype(np.uint8)
    _, BW = cv2.threshold(gray_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    BW = BW / 255.0
    
    foreground_ratio = np.sum(BW) / BW.size
    
    # 连通域数量
    BW_uint8 = (BW * 255).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(BW_uint8)
    num_objects = num_labels - 1
    
    # 边缘密度 (Sobel)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = convolve(gray, sobel_x, mode='reflect')
    gy = convolve(gray, sobel_y, mode='reflect')
    edges = np.sqrt(gx**2 + gy**2)
    edge_density = np.sum(edges > 0.1) / edges.size
    
    # 局部方差
    local_var = cv2.GaussianBlur(gray**2, (5,5), 1.0) - cv2.GaussianBlur(gray, (5,5), 1.0)**2
    mean_var = np.mean(local_var[BW > 0]) if np.sum(BW > 0) > 0 else 0
    
    is_discrete = (foreground_ratio > 0.05 and foreground_ratio < 0.5) and \
                  (num_objects > 20) and \
                  (edge_density > 0.05 and edge_density < 0.4) and \
                  (mean_var > 0.05)
    
    print(f"  [图像分析] 前景比例:{foreground_ratio*100:.1f}%, 对象数:{num_objects}, 边缘密度:{edge_density:.2f}, 判定:{'离散型图像' if is_discrete else '通用图像'}")
    return is_discrete

# ==================== 4. 多尺度修正拉普拉斯 ====================
def multi_scale_modified_laplacian(I, step_list):
    """对应 MATLAB 的 multi_scale_modified_laplacian"""
    I = I.astype(np.float32)
    h, w = I.shape
    F = np.zeros((h, w), dtype=np.float32)
    for step in step_list:
        I_pad = np.pad(I, step, mode='symmetric')
        center = I_pad[step:step+h, step:step+w]
        left = I_pad[step:step+h, :w]
        right = I_pad[step:step+h, 2*step:2*step+w]
        up = I_pad[:h, step:step+w]
        down = I_pad[2*step:2*step+h, step:step+w]
        term1 = np.abs(2*center - left - right)
        term2 = np.abs(2*center - up - down)
        F += term1 + term2
    return F / len(step_list)

# ==================== 5. 空间频率 ====================
def spatial_frequency(I):
    """对应 MATLAB 的 spatial_frequency"""
    I = I.astype(np.float32)
    # 水平差分核
    dx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float32)
    # 垂直差分核
    dy = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)
    
    dI_x = convolve(I, dx, mode='reflect')
    dI_y = convolve(I, dy, mode='reflect')
    
    dI_x2 = dI_x**2
    dI_y2 = dI_y**2
    
    win_size = 7
    kernel = np.ones((win_size, win_size), dtype=np.float32) / (win_size**2)
    local_x2 = convolve(dI_x2, kernel, mode='reflect')
    local_y2 = convolve(dI_y2, kernel, mode='reflect')
    
    SF = np.sqrt(local_x2 + local_y2)
    # 归一化到 [0,1]
    SF = (SF - np.min(SF)) / (np.max(SF) - np.min(SF) + 1e-8)
    return SF

# ==================== 6. 梯度幅度 (Tenengrad) ====================
def gradient_magnitude(I):
    """对应 MATLAB 的 gradient_magnitude"""
    I = I.astype(np.float32)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    Gx = convolve(I, sobel_x, mode='reflect')
    Gy = convolve(I, sobel_y, mode='reflect')
    
    GM = np.sqrt(Gx**2 + Gy**2)
    
    win_size = 7
    kernel = np.ones((win_size, win_size), dtype=np.float32) / (win_size**2)
    GM_local = convolve(GM, kernel, mode='reflect')
    
    GM = (GM_local - np.min(GM_local)) / (np.max(GM_local) - np.min(GM_local) + 1e-8)
    return GM

# ==================== 7. 多尺度空间频率 ====================
def multi_scale_spatial_frequency(I, scale_list):
    """对应 MATLAB 的 multi_scale_spatial_frequency"""
    I = I.astype(np.float32)
    H, W = I.shape
    F_SF = np.zeros((H, W), dtype=np.float32)
    for scale in scale_list:
        if scale == 1:
            SF_map = spatial_frequency(I)
        else:
            new_h = int(H / scale)
            new_w = int(W / scale)
            I_scaled = cv2.resize(I, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            SF_scaled = spatial_frequency(I_scaled)
            SF_map = cv2.resize(SF_scaled, (W, H), interpolation=cv2.INTER_LINEAR)
        F_SF += SF_map
    F_SF = F_SF / len(scale_list)
    F_SF = (F_SF - np.min(F_SF)) / (np.max(F_SF) - np.min(F_SF) + 1e-8)
    return F_SF

# ==================== 8. 多特征聚焦度量 ====================
def multi_feature_focus_measure(I, step_list, w_mml=0.5, w_sf=0.3, w_gm=0.2):
    """对应 MATLAB 的 multi_feature_focus_measure"""
    I = I.astype(np.float32)
    
    F_mml = multi_scale_modified_laplacian(I, step_list)
    F_mml = (F_mml - np.min(F_mml)) / (np.max(F_mml) - np.min(F_mml) + 1e-8)
    
    F_sf = multi_scale_spatial_frequency(I, [1, 2])
    F_sf = (F_sf - np.min(F_sf)) / (np.max(F_sf) - np.min(F_sf) + 1e-8)
    
    F_gm = gradient_magnitude(I)
    F_gm = (F_gm - np.min(F_gm)) / (np.max(F_gm) - np.min(F_gm) + 1e-8)
    
    FM = w_mml * F_mml + w_sf * F_sf + w_gm * F_gm
    FM = (FM - np.min(FM)) / (np.max(FM) - np.min(FM) + 1e-8)
    return FM

# ==================== 9. NDF 结构层 ====================
def NDF_StructureLayer(img, win=7):
    """对应 MATLAB 的 NDF_StructureLayer"""
    img = img.astype(np.float32)
    H, W, C = img.shape
    r = int(win / 2)
    S = np.zeros_like(img)
    for c in range(C):
        S[:, :, c] = cv2.GaussianBlur(img[:, :, c], (win, win), r/2, borderType=cv2.BORDER_REFLECT)
    D = img - S
    S = (S - np.min(S)) / (np.max(S) - np.min(S) + 1e-8)
    return S, D

# ==================== 10. BCV 优化 ====================
def bcv_optimization(img, bsz, nsz):
    """对应 MATLAB 的 bcv_optimization"""
    img = img.astype(np.float32)
    h, w = img.shape
    num_h = int(np.ceil(h / bsz))
    num_w = int(np.ceil(w / bsz))
    
    block_focus_ratio = np.zeros((num_h, num_w), dtype=np.float32)
    for i in range(num_h):
        for j in range(num_w):
            row_start = i * bsz
            row_end = min((i+1) * bsz, h)
            col_start = j * bsz
            col_end = min((j+1) * bsz, w)
            block = img[row_start:row_end, col_start:col_end]
            block_focus_ratio[i, j] = np.sum(block) / block.size
    
    # 平均滤波
    kernel = np.ones((nsz, nsz), dtype=np.float32) / (nsz**2)
    block_update = convolve(block_focus_ratio, kernel, mode='reflect')
    
    # 双线性插值
    Xq = np.linspace(0, w-1, num_w)
    Yq = np.linspace(0, h-1, num_h)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    from scipy.interpolate import RegularGridInterpolator
    interpolator = RegularGridInterpolator((Yq, Xq), block_update, method='linear', bounds_error=False, fill_value=0.5)
    points = np.stack([Y.ravel(), X.ravel()], axis=-1)
    img_bcv = interpolator(points).reshape(h, w)
    
    img_bcv = np.clip(img_bcv, 0, 1)
    img_bcv = np.nan_to_num(img_bcv, nan=0.5)
    return img_bcv

def generate_mid_decision_map_bcv(IDI1, IDI2, block_size, nbr_size):
    """对应 MATLAB 的 generate_mid_decision_map_bcv"""
    MDI1 = bcv_optimization(IDI1, block_size, nbr_size)
    MDI2 = bcv_optimization(IDI2, block_size, nbr_size)
    
    MDI_sum = MDI1 + MDI2 + 1e-8
    MDI1 = MDI1 / MDI_sum
    MDI2 = MDI2 / MDI_sum
    return MDI1, MDI2

# ==================== 11. 引导滤波 ====================
def guided_filtering(guide, p, r, eps):
    """对应 MATLAB 的 guided_filtering"""
    guide = guide.astype(np.float32)
    p = p.astype(np.float32)
    
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT) / ((2*r+1)**2)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT) / ((2*r+1)**2)
    
    corr_I = cv2.boxFilter(guide*guide, cv2.CV_32F, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT) / ((2*r+1)**2)
    corr_Ip = cv2.boxFilter(guide*p, cv2.CV_32F, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT) / ((2*r+1)**2)
    
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT) / ((2*r+1)**2)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (2*r+1, 2*r+1), borderType=cv2.BORDER_REFLECT) / ((2*r+1)**2)
    
    out = mean_a * guide + mean_b
    return np.clip(out, 0, 1)

def guided_filter_decision_map(MDI1, MDI2, I_guide, r, eps_val):
    """对应 MATLAB 的 guided_filter_decision_map"""
    MDI1 = MDI1.astype(np.float32)
    MDI2 = MDI2.astype(np.float32)
    I_guide = I_guide.astype(np.float32)
    
    FDI1 = guided_filtering(I_guide, MDI1, r, eps_val)
    FDI2 = guided_filtering(I_guide, MDI2, r, eps_val)
    
    FDI_sum = FDI1 + FDI2 + 1e-8
    FDI1 = FDI1 / FDI_sum
    FDI2 = FDI2 / FDI_sum
    return FDI1, FDI2

# ==================== 12. 焦点边界检测 ====================
def detect_focus_boundary(MDI1, MDI2, threshold):
    """对应 MATLAB 的 detect_focus_boundary"""
    diff_map = np.abs(MDI1 - MDI2)
    
    kernel = np.ones((3,3), dtype=np.float32)
    grad_MDI1 = cv2.dilate(MDI1, kernel) - cv2.erode(MDI1, kernel)
    grad_MDI2 = cv2.dilate(MDI2, kernel) - cv2.erode(MDI2, kernel)
    
    boundary_strength = diff_map + 0.5 * (grad_MDI1 + grad_MDI2)
    boundary_mask = (boundary_strength > threshold) & (boundary_strength < 0.9)
    
    weights = np.zeros_like(MDI1)
    if np.any(boundary_mask):
        weights[boundary_mask] = MDI1[boundary_mask] / (MDI1[boundary_mask] + MDI2[boundary_mask] + 1e-8)
        weights = cv2.GaussianBlur(weights, (5,5), 1.5, borderType=cv2.BORDER_REFLECT)
        weights = np.clip(weights, 0, 1)
    return boundary_mask, weights

# ==================== 13. 聚焦质量图 ====================
def compute_focus_map(img):
    """对应 MATLAB 的 compute_focus_map"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = gray.astype(np.float32) / 255.0
    focus_map = multi_feature_focus_measure(gray, [3, 5, 7], 0.5, 0.3, 0.2)
    return focus_map

# ==================== 14. 置信掩码 ====================
def get_confidence_mask(focus_map, conf_thresh):
    """对应 MATLAB 的 get_confidence_mask"""
    return focus_map > conf_thresh

# ==================== 15. 聚焦权重 ====================
def focus2weight(focus, min_w=0.1, max_w=0.9):
    """对应 MATLAB 的 focus2weight"""
    focus = (focus - np.min(focus)) / (np.max(focus) - np.min(focus) + 1e-8)
    weight = min_w + (max_w - min_w) * focus
    return weight

# ==================== 16. 质量控制 ====================
def check_fusion_quality(I_before, I_after, focus_mask, threshold=0.85):
    """对应 MATLAB 的 check_fusion_quality"""
    if len(I_before.shape) == 3:
        gray_before = cv2.cvtColor(I_before, cv2.COLOR_BGR2GRAY)
    else:
        gray_before = I_before
    if len(I_after.shape) == 3:
        gray_after = cv2.cvtColor(I_after, cv2.COLOR_BGR2GRAY)
    else:
        gray_after = I_after
    
    gray_before = gray_before.astype(np.float32) / 255.0
    gray_after = gray_after.astype(np.float32) / 255.0
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    gx_before = convolve(gray_before, sobel_x, mode='reflect')
    gy_before = convolve(gray_before, sobel_y, mode='reflect')
    gx_after = convolve(gray_after, sobel_x, mode='reflect')
    gy_after = convolve(gray_after, sobel_y, mode='reflect')
    
    grad_before = np.sqrt(gx_before**2 + gy_before**2)
    grad_after = np.sqrt(gx_after**2 + gy_after**2)
    
    if np.any(focus_mask):
        mean_grad_before = np.mean(grad_before[focus_mask])
        mean_grad_after = np.mean(grad_after[focus_mask])
        degradation_ratio = mean_grad_after / (mean_grad_before + 1e-8)
    else:
        degradation_ratio = 1.0
    
    quality_ok = degradation_ratio >= threshold
    return quality_ok, degradation_ratio

# ==================== 17. 评价指标 ====================
def ssim_metric(A, F):
    """对应 MATLAB 的 ssim_metric"""
    if len(A.shape) == 3:
        A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    if len(F.shape) == 3:
        F = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
    A = A.astype(np.float32) / 255.0
    F = F.astype(np.float32) / 255.0
    
    C1 = (0.01 * 1)**2
    C2 = (0.03 * 1)**2
    
    kernel = cv2.getGaussianKernel(11, 1.5)
    kernel = kernel @ kernel.T
    
    muA = cv2.filter2D(A, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    muF = cv2.filter2D(F, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    
    sigmaA2 = cv2.filter2D(A**2, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT) - muA**2
    sigmaF2 = cv2.filter2D(F**2, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT) - muF**2
    sigmaAF = cv2.filter2D(A*F, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT) - muA*muF
    
    L = (2*muA*muF + C1) / (muA**2 + muF**2 + C1)
    C = (2*np.sqrt(sigmaA2)*np.sqrt(sigmaF2) + C2) / (sigmaA2 + sigmaF2 + C2)
    S = (sigmaAF + C2/2) / (np.sqrt(sigmaA2)*np.sqrt(sigmaF2) + C2/2)
    
    ssim_val = np.mean(L * C * S)
    return float(ssim_val)

def mutual_info_metric(A, B, F):
    """对应 MATLAB 的 mutual_info_metric"""
    if len(A.shape) == 3:
        A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    if len(B.shape) == 3:
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    if len(F.shape) == 3:
        F = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
    
    A = A.astype(np.float32) / 255.0
    B = B.astype(np.float32) / 255.0
    F = F.astype(np.float32) / 255.0
    
    def mi_single(X, Y):
        X = (X * 255).astype(np.uint8)
        Y = (Y * 255).astype(np.uint8)
        h, _, _ = np.histogram2d(X.ravel(), Y.ravel(), bins=64)
        p = h / (np.sum(h) + 1e-8)
        px = np.sum(p, axis=1)
        py = np.sum(p, axis=0)
        Hx = -np.sum(px[px > 0] * np.log2(px[px > 0] + 1e-8))
        Hy = -np.sum(py[py > 0] * np.log2(py[py > 0] + 1e-8))
        Hxy = -np.sum(p[p > 0] * np.log2(p[p > 0] + 1e-8))
        mi = Hx + Hy - Hxy
        return mi
    
    miAF = mi_single(A, F)
    miBF = mi_single(B, F)
    mi_val = (miAF + miBF) / 2
    return float(mi_val)

def qabf_metric(A, B, F):
    """对应 MATLAB 的 qabf_metric"""
    if len(A.shape) == 3:
        A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    if len(B.shape) == 3:
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    if len(F.shape) == 3:
        F = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    F = F.astype(np.float32)
    
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    gA = np.sqrt(convolve(A, sobel_x, mode='reflect')**2 + convolve(A, sobel_y, mode='reflect')**2)
    gB = np.sqrt(convolve(B, sobel_x, mode='reflect')**2 + convolve(B, sobel_y, mode='reflect')**2)
    gF = np.sqrt(convolve(F, sobel_x, mode='reflect')**2 + convolve(F, sobel_y, mode='reflect')**2)
    
    wA = gA / (gA + gB + 1e-8)
    wB = gB / (gA + gB + 1e-8)
    
    QAF = (2 * gA * gF + 1e-8) / (gA**2 + gF**2 + 1e-8)
    QBF = (2 * gB * gF + 1e-8) / (gB**2 + gF**2 + 1e-8)
    q_map = wA * QAF + wB * QBF
    qabf_val = np.mean(q_map)
    return float(qabf_val)

def qm_metric(A, B, F):
    """对应 MATLAB 的 qm_metric"""
    if len(A.shape) == 3:
        A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    if len(B.shape) == 3:
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    if len(F.shape) == 3:
        F = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
    
    scores = []
    for scale in range(3):
        # 使用 Sobel 边缘检测
        gA = cv2.Sobel(A, cv2.CV_32F, 1, 1, ksize=3)
        gB = cv2.Sobel(B, cv2.CV_32F, 1, 1, ksize=3)
        gF = cv2.Sobel(F, cv2.CV_32F, 1, 1, ksize=3)
        gA = np.abs(gA) > 0.1 * np.max(gA)
        gB = np.abs(gB) > 0.1 * np.max(gB)
        gF = np.abs(gF) > 0.1 * np.max(gF)
        
        T1 = np.sum(gF & gA) / (np.sum(gA) + 1e-8)
        T2 = np.sum(gF & gB) / (np.sum(gB) + 1e-8)
        scores.append((T1 + T2) / 2)
        
        A = cv2.resize(A, (A.shape[1]//2, A.shape[0]//2), interpolation=cv2.INTER_LINEAR)
        B = cv2.resize(B, (B.shape[1]//2, B.shape[0]//2), interpolation=cv2.INTER_LINEAR)
        F = cv2.resize(F, (F.shape[1]//2, F.shape[0]//2), interpolation=cv2.INTER_LINEAR)
    
    qm_val = np.mean(scores)
    return float(qm_val)

def qcb_metric(A, B, F):
    """对应 MATLAB 的 qcb_metric"""
    if len(A.shape) == 3:
        A = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    if len(B.shape) == 3:
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    if len(F.shape) == 3:
        F = cv2.cvtColor(F, cv2.COLOR_BGR2GRAY)
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    F = F.astype(np.float32)
    
    kernel = cv2.getGaussianKernel(5, 1.0)
    kernel = kernel @ kernel.T
    
    A_csf = cv2.filter2D(A, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    B_csf = cv2.filter2D(B, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    F_csf = cv2.filter2D(F, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
    
    muA = np.mean(A_csf)
    muB = np.mean(B_csf)
    muF = np.mean(F_csf)
    sigmaA = np.std(A_csf)
    sigmaB = np.std(B_csf)
    sigmaF = np.std(F_csf)
    
    QAF = (4 * muA * muF * sigmaA * sigmaF + 1e-8) / ((muA**2 + muF**2) * (sigmaA**2 + sigmaF**2) + 1e-8)
    QBF = (4 * muB * muF * sigmaB * sigmaF + 1e-8) / ((muB**2 + muF**2) * (sigmaB**2 + sigmaF**2) + 1e-8)
    qcb_val = (QAF + QBF) / 2
    return float(qcb_val)

# ==================== 18. 主融合函数 ====================
def fuse_images(images):
    """主融合函数 - 严格对应 MATLAB 的 image_fusion_algorithm"""
    if len(images) < 2:
        return images[0]
    
    print(f"📷 成功选择 {len(images)} 张图像，开始处理...")
    num_images = len(images)
    
    # ===== 图像读取与预处理 =====
    print("\n========== 图像读取 ==========")
    
    # 统一尺寸（使用第一张作为参考）
    I_ref = images[0]
    if len(I_ref.shape) == 2:
        I_ref = cv2.cvtColor(I_ref, cv2.COLOR_GRAY2BGR)
    ref_H, ref_W = I_ref.shape[:2]
    print(f"参考图像尺寸: {ref_H}x{ref_W}")
    
    # 统一所有图像尺寸
    imgs = []
    for i, img in enumerate(images):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[:2] != (ref_H, ref_W):
            img = cv2.resize(img, (ref_W, ref_H), interpolation=cv2.INTER_LINEAR)
        imgs.append(img)
        print(f"图像{i+1}: 已调整尺寸")
    
    # ===== 背景优化预处理 =====
    print("\n========== 背景优化预处理 ==========")
    images_opt = []
    for i, img in enumerate(imgs):
        img_opt = background_optimization_core(img, 2)
        images_opt.append(img_opt)
        print(f"图像{i+1} 背景优化完成")
    
    # ===== 图像类型检测 =====
    print("\n========== 图像类型检测 ==========")
    is_discrete = detect_discrete_image(images_opt[0])
    if is_discrete:
        print("检测到离散型图像（如小球藻），使用优化参数...")
    
    # ===== 参数设置 =====
    if is_discrete:
        conf_thresh = 0.55
        update_delta = 0.02
        weak_delta = 0.005
        win_sizes = [1, 2, 3]
        ndf_win = 3
        bcv_block_size = 6
        bcv_nbr_size = 3
        guide_r = 4
        guide_eps = 0.005
        boundary_thresh = 0.08
        smooth_sigma = 1.5
        print(f"离散图像模式：win={win_sizes}, ndf={ndf_win}, bcv={bcv_block_size}, guide={guide_r}")
    else:
        conf_thresh = 0.65
        update_delta = 0.03
        weak_delta = 0.01
        win_sizes = [3, 5, 7]
        ndf_win = 7
        bcv_block_size = 12
        bcv_nbr_size = 7
        guide_r = 8
        guide_eps = 0.01
        boundary_thresh = 0.1
        smooth_sigma = 2.5
        print(f"通用图像模式：win={win_sizes}, ndf={ndf_win}, bcv={bcv_block_size}, guide={guide_r}")
    
    # ===== 迭代融合 =====
    print("\n========== 分层融合 + 增强聚焦记忆 ==========")
    print("使用优化配置：多特征聚焦度量 + 引导滤波 + 边界智能处理")
    
    fused_result = images_opt[0].copy()
    best_focus_map = compute_focus_map(fused_result)
    conf_protect_mask = get_confidence_mask(best_focus_map, conf_thresh)
    
    for i in range(1, num_images):
        print(f"第 {i}/{num_images-1} 次融合：融合结果 + 图像{i+1}")
        I_curr = fused_result
        I_new = images_opt[i]
        
        # 计算聚焦图
        new_focus_map = compute_focus_map(I_new)
        strong_update = new_focus_map > best_focus_map + update_delta
        weak_update = (new_focus_map > best_focus_map + weak_delta) & ~strong_update
        total_update = (strong_update | weak_update) & ~conf_protect_mask
        
        if not np.any(total_update):
            print("  无有效更新区域，跳过本次融合")
            continue
        
        # 灰度图
        I_curr_gray = cv2.cvtColor(I_curr, cv2.COLOR_BGR2GRAY)
        I_new_gray = cv2.cvtColor(I_new, cv2.COLOR_BGR2GRAY)
        I_curr_gray = I_curr_gray.astype(np.float32) / 255.0
        I_new_gray = I_new_gray.astype(np.float32) / 255.0
        
        # 多特征聚焦度量
        F1 = multi_feature_focus_measure(I_curr_gray, win_sizes, 0.5, 0.3, 0.2)
        F2 = multi_feature_focus_measure(I_new_gray, win_sizes, 0.5, 0.3, 0.2)
        
        # 初始决策图
        IDI1 = F1 / (F1 + F2 + 1e-8)
        IDI2 = F2 / (F1 + F2 + 1e-8)
        
        # BCV 中间决策图
        MDI1, MDI2 = generate_mid_decision_map_bcv(IDI1, IDI2, bcv_block_size, bcv_nbr_size)
        
        # 引导滤波决策图
        I_guide = I_curr_gray
        FDI1, FDI2 = guided_filter_decision_map(MDI1, MDI2, I_guide, guide_r, guide_eps)
        
        # 扩展为三通道
        F1L = np.stack([FDI1, FDI1, FDI1], axis=2)
        F2L = np.stack([FDI2, FDI2, FDI2], axis=2)
        F1H = F1L
        F2H = F2L
        
        # NDF 分层分解
        S_curr, D_curr = NDF_StructureLayer(I_curr, ndf_win)
        S_new, D_new = NDF_StructureLayer(I_new, ndf_win)
        
        # 分层融合
        fused_S = S_curr * F1L + S_new * F2L
        fused_D = D_curr * F1H + D_new * F2H
        fused_layer = fused_S + fused_D
        fused_layer = np.clip(fused_layer, 0, 1)
        
        # 焦点边界智能处理
        boundary_mask, boundary_weights = detect_focus_boundary(FDI1, FDI2, boundary_thresh)
        if np.any(boundary_mask):
            print(f"  检测到边界区域（{np.mean(boundary_mask)*100:.2f}%像素），使用智能加权融合")
            bw_3ch = np.stack([boundary_weights, boundary_weights, boundary_weights], axis=2)
            FDI1_3ch = np.stack([FDI1, FDI1, FDI1], axis=2)
            FDI2_3ch = np.stack([FDI2, FDI2, FDI2], axis=2)
            fused_layer = fused_layer * (1 - bw_3ch) + \
                          (I_curr * FDI1_3ch + I_new * FDI2_3ch) * bw_3ch
            fused_layer = np.clip(fused_layer, 0, 1)
        
        # 掩码平滑过渡
        smooth_sigma_local = 1.5 if is_discrete else 2.5
        update_mask = total_update.astype(np.float32)
        update_mask = cv2.GaussianBlur(update_mask, (5,5), smooth_sigma_local, borderType=cv2.BORDER_REFLECT)
        update_mask = np.stack([update_mask, update_mask, update_mask], axis=2)
        
        fused_temp = (1 - update_mask) * I_curr + update_mask * fused_layer
        fused_temp = np.clip(fused_temp, 0, 1)
        
        # 质量控制
        clear_region_mask = conf_protect_mask
        quality_ok, degradation_ratio = check_fusion_quality(I_curr, fused_temp, clear_region_mask, 0.85)
        
        if not quality_ok:
            print(f"  [质量控制] 融合质量不合格（梯度保留率 {degradation_ratio*100:.1f}%），跳过本次融合")
            continue
        
        # 更新融合结果
        fused_result = fused_temp
        best_focus_map = np.maximum(best_focus_map, new_focus_map)
        conf_protect_mask = get_confidence_mask(best_focus_map, conf_thresh)
        
        update_ratio = np.mean(total_update) * 100
        print(f"  融合完成，有效更新像素比例: {update_ratio:.2f}%，质量合格（梯度保留 {degradation_ratio*100:.1f}%）")
    
    fused_result = np.clip(fused_result, 0, 1)
    return (fused_result * 255).astype(np.uint8)
