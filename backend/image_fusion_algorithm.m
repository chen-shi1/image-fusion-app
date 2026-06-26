function fused_result = main_fusion(images_cell)

close all; clear; clc;

% 功能：多图像序列融合系统（背景优化 + 分层融合 + 增强聚焦记忆 + 清晰区域保护）
% 修复问题：迭代融合中原有清晰区域被后续模糊区域覆盖
% 
% 优化版本新增：
%   1. 多特征聚焦度量融合（MML + 空间频率 + 梯度幅度）
%   2. 引导滤波决策图优化（替代盒式滤波）
%   3. 焦点边界智能处理（FDB检测 + 加权插值）

%% ===================== 函数定义区 =====================

% ==================== 小球藻离散图像检测模块 ====================
% 功能：自动检测图像是否为离散型图像（如小球藻）
% 离散图像特点：前景对象小且分离、背景均匀、对象边界清晰
function is_discrete = detect_discrete_image(I)
    I = im2double(rgb2gray(I));
    
    % 检测前景占比
    threshold = graythresh(I);
    BW = imbinarize(I, threshold);
    foreground_ratio = sum(BW(:)) / numel(BW);
    
    % 检测对象数量（连通域）
    CC = bwconncomp(BW);
    num_objects = CC.NumObjects;
    
    % 检测边缘密度
    edges = edge(I, 'sobel');
    edge_density = sum(edges(:)) / numel(edges);
    
    % 检测局部方差（离散图像前景区域方差较大）
    local_var = stdfilt(I);
    mean_var = mean(local_var(BW));
    
    % 判断条件：前景占比适中(5%-50%)、对象数量>20、边缘密度适中
    is_discrete = (foreground_ratio > 0.05 && foreground_ratio < 0.5) && ...
                  (num_objects > 20) && ...
                  (edge_density > 0.05 && edge_density < 0.4) && ...
                  (mean_var > 0.05);
              
    fprintf('  [图像分析] 前景比例:%.1f%%, 对象数:%d, 边缘密度:%.2f, 判定:%s\n', ...
            foreground_ratio*100, num_objects, edge_density, ...
            ternaryOp(is_discrete, '离散型图像', '通用图像'));
end

function result = ternaryOp(condition, trueVal, falseVal)
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end

% ------------- 函数1：背景优化模块（输出double [0,1]）- 优化版 -------------
% 基于最新研究MFusionJ (2026)：使用边缘保持滤波(EPF)替代传统高斯滤波
function info_img = background_optimization_core(I, sigma)
    if nargin < 2
        sigma = 2;
    end
    % 转换为double并保持范围[0,1]
    I_double = im2double(I);
    HSV = rgb2hsv(I_double);
    S = HSV(:, :, 2);
    
    % 使用自适应阈值分割（改进版）
    threshold = graythresh(S);
    BW = imbinarize(S, threshold);
    
    % 形态学操作优化背景区域检测（基于最新研究）
    % 使用形态学开运算去除小噪声区域
    BW = imopen(BW, ones(3,3));
    
    % 边缘保持滤波替代传统高斯滤波（EPF优化）
    % 使用双边滤波保留边缘信息
    for c = 1:3
        channel = I_double(:, :, c);
        % 对背景区域使用边缘保持滤波
        % 双边滤波参数：空间sigma和颜色sigma
        channel_smooth = bilateral_filter(channel, sigma, sigma*0.1);
        channel(BW == 0) = channel_smooth(BW == 0);
        info_img(:, :, c) = channel;
    end
    % 确保输出为double [0,1]
    info_img = mat2gray(info_img);
end

% ------------- 辅助函数：双边滤波（边缘保持滤波） -------------
function J = bilateral_filter(I, sigma_s, sigma_r)
    % 简化版双边滤波实现
    % sigma_s: 空域标准差
    % sigma_r: 像素值域标准差
    
    if nargin < 3
        sigma_r = sigma_s * 0.1;
    end
    
    % 使用MATLAB内置函数（如果可用）
    try
        J = imbilatfilt(I, sigma_s, sigma_r);
    catch
        % 手动实现简化版双边滤波
        win_size = ceil(3 * sigma_s);
        if mod(win_size, 2) == 0
            win_size = win_size + 1;
        end
        half_win = floor(win_size / 2);
        
        [H, W] = size(I);
        J = zeros(H, W);
        
        % 预计算空间权重
        [x, y] = meshgrid(-half_win:half_win, -half_win:half_win);
        spatial_weight = exp(-(x.^2 + y.^2) / (2 * sigma_s^2));
        
        % 对每个像素进行滤波
        I_pad = padarray(I, [half_win, half_win], 'symmetric');
        for i = 1:H
            for j = 1:W
                center_val = I_pad(i+half_win, j+half_win);
                block = I_pad(i:i+win_size-1, j:j+win_size-1);
                
                % 计算像素值权重
                range_weight = exp(-(block - center_val).^2 / (2 * sigma_r^2));
                
                % 组合权重
                total_weight = spatial_weight .* range_weight;
                total_weight = total_weight / sum(total_weight(:));
                
                % 滤波结果
                J(i, j) = sum(sum(block .* total_weight));
            end
        end
    end
end

% ------------- 函数2：邻距滤波器（改进版，三通道独立滤波）- 优化版 -------------
% 优化：增加向量化实现版本，减少循环提升速度
function [S, D] = NDF_StructureLayer(img, win, use_vectorized)
    % 输入 img: double RGB [0,1], win: 滤波窗口大小（奇数）
    % use_vectorized: 是否使用向量化实现（默认true，速度更快）
    if nargin < 2
        win = 7;
    end
    if nargin < 3
        use_vectorized = true;
    end
    
    img = im2double(img);
    [H, W, C] = size(img);
    r = floor(win / 2);
    
    if use_vectorized && ~isempty(which('imgaussfilt'))
        % 使用向量化实现（快速版本）
        S = zeros(H, W, C);
        for c = 1:C
            % 使用imgaussfilt进行高斯滤波
            S(:, :, c) = imgaussfilt(img(:, :, c), r/2, 'Padding', 'symmetric');
        end
    else
        % 传统循环实现（兼容性版本）
        [x, y] = meshgrid(-r:r, -r:r);
        weight = exp(-(x.^2 + y.^2) / (2 * (r/2)^2));
        weight = weight / sum(weight(:));
        
        S = zeros(H, W, C);
        for c = 1:C
            channel = img(:, :, c);
            channel_pad = padarray(channel, [r, r], 'symmetric');
            S_channel = zeros(H, W);
            for i = r+1 : H+r
                for j = r+1 : W+r
                    block = channel_pad(i-r:i+r, j-r:j+r);
                    S_channel(i-r, j-r) = sum(block(:) .* weight(:));
                end
            end
            S(:, :, c) = S_channel;
        end
    end
    
    % 细节层：原始图像 - 结构层
    D = img - S;
    % 结构层归一化到[0,1]
    S = mat2gray(S);
end

% ------------- 函数3：多尺度修正拉普拉斯算子（优化版） -------------
% 优化：增加向量化实现，减少循环
function F = multi_scale_modified_laplacian(I, step_list)
    I = im2double(I);
    [H, W] = size(I);
    F = zeros(H, W);
    num_steps = length(step_list);
    
    for step = step_list
        I_pad = padarray(I, [step, step], 'symmetric');
        
        % 向量化计算：避免逐像素循环
        center = I_pad(step+1:end-step, step+1:end-step);
        left   = I_pad(step+1:end-step, 1:end-2*step);
        right  = I_pad(step+1:end-step, 2*step+1:end);
        up     = I_pad(1:end-2*step, step+1:end-step);
        down   = I_pad(2*step+1:end, step+1:end-step);
        
        term1 = abs(2*center - left - right);
        term2 = abs(2*center - up - down);
        ML = term1 + term2;
        
        F = F + ML;
    end
    F = F / num_steps;
end

% ------------- 函数5：中间决策图生成（BCV分块插值）- 修复版输出连续值 -------------
function [MDI1, MDI2] = generate_mid_decision_map_bcv(IDI1, IDI2, block_size, nbr_size)
    MDI1 = bcv_optimization(IDI1, block_size, nbr_size);
    MDI2 = bcv_optimization(IDI2, block_size, nbr_size);
    
    % 确保MDI1和MDI2互补（归一化）
    MDI_sum = MDI1 + MDI2 + eps;
    MDI1 = MDI1 ./ MDI_sum;
    MDI2 = MDI2 ./ MDI_sum;
    
    function img_bcv = bcv_optimization(img, bsz, nsz)
        [h, w] = size(img);
        img = im2double(img);
        num_h = ceil(h / bsz);
        num_w = ceil(w / bsz);
        block_focus_ratio = zeros(num_h, num_w);
        for i = 1:num_h
            for j = 1:num_w
                row_start = (i-1)*bsz + 1;
                row_end = min(i*bsz, h);
                col_start = (j-1)*bsz + 1;
                col_end = min(j*bsz, w);
                block = img(row_start:row_end, col_start:col_end);
                block_focus_ratio(i,j) = sum(block(:)) / numel(block);
            end
        end
        % 使用连续值而非二值化
        block_update = imfilter(block_focus_ratio, fspecial('average', nsz), 'symmetric');
        [X, Y] = meshgrid(1:w, 1:h);
        [Xq, Yq] = meshgrid(linspace(1,w,num_w), linspace(1,h,num_h));
        % 插值得到连续决策图
        img_bcv = interp2(Xq, Yq, block_update, X, Y, 'linear');
        % 确保值在[0,1]范围内
        img_bcv = max(0, min(1, img_bcv));
        % 处理NaN值（边界区域）
        img_bcv(isnan(img_bcv)) = 0.5;
    end
end

% ------------- 函数7：聚焦质量图计算（用于记忆机制）- 优化版使用多特征融合 -------------
function focus_map = compute_focus_map(img)
    gray = rgb2gray(im2double(img));
    % 使用多特征融合聚焦度量（MML + 空间频率 + 梯度幅度）
    focus_map = multi_feature_focus_measure(gray, [3,5,7], 0.5, 0.3, 0.2);
    focus_map = mat2gray(focus_map);
end

% ------------- 新增1：生成像素置信掩码（锁定高清晰区域，禁止覆盖） -------------
function conf_mask = get_confidence_mask(focus_map, conf_thresh)
    conf_mask = focus_map > conf_thresh;
end

% ------------- 新增2：聚焦度归一化权重（清晰度映射为融合权重） -------------
function weight = focus2weight(focus, min_w, max_w)
    if nargin < 3
        min_w = 0.1;
        max_w = 0.9;
    end
    focus = mat2gray(focus);
    weight = min_w + (max_w - min_w) .* focus;
end

% ------------- 新增3：质量控制函数（检测融合后清晰区域是否模糊） -------------
% 功能：比较融合前后清晰区域的梯度值或清晰度评价指标
% 返回：true表示质量合格，false表示质量下降超过阈值（低于85%）
function [quality_ok, degradation_ratio] = check_fusion_quality(I_before, I_after, focus_mask, threshold)
    if nargin < 4
        threshold = 0.85;  % 默认85%阈值
    end
    
    % 转换为灰度图
    gray_before = rgb2gray(im2double(I_before));
    gray_after = rgb2gray(im2double(I_after));
    
    % 计算清晰区域的梯度值（使用Tenengrad梯度，基于最新研究）
    % Tenengrad梯度：Sobel滤波器的平方和
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    
    gx_before = imfilter(gray_before, sobel_x, 'symmetric');
    gy_before = imfilter(gray_before, sobel_y, 'symmetric');
    gx_after = imfilter(gray_after, sobel_x, 'symmetric');
    gy_after = imfilter(gray_after, sobel_y, 'symmetric');
    
    grad_before = sqrt(gx_before.^2 + gy_before.^2);
    grad_after = sqrt(gx_after.^2 + gy_after.^2);
    
    % 计算清晰区域的平均梯度值
    if any(focus_mask(:))
        mean_grad_before = mean(grad_before(focus_mask));
        mean_grad_after = mean(grad_after(focus_mask));
    else
        % 如果没有清晰区域，返回合格
        quality_ok = true;
        degradation_ratio = 1.0;
        return;
    end
    
    % 计算退化比例
    degradation_ratio = mean_grad_after / mean_grad_before;
    
    % 判断是否低于阈值
    quality_ok = degradation_ratio >= threshold;
    
    if ~quality_ok
        fprintf('  [质量警告] 清晰区域梯度下降 %.1f%%，低于85%%阈值，跳过本次融合\n', (1-degradation_ratio)*100);
    end
end

% ------------- 原始评价指标函数 -------------
function ssim_val = ssim_metric(A, F)
    A = rgb2gray(im2double(A));
    F = rgb2gray(im2double(F));
    C1 = (0.01 * 1)^2; C2 = (0.03 * 1)^2;
    gauss_filter = fspecial('gaussian', 11, 1.5);
    muA = imfilter(A, gauss_filter, 'symmetric');
    muF = imfilter(F, gauss_filter, 'symmetric');
    sigmaA2 = imfilter(A.^2, gauss_filter, 'symmetric') - muA.^2;
    sigmaF2 = imfilter(F.^2, gauss_filter, 'symmetric') - muF.^2;
    sigmaAF = imfilter(A.*F, gauss_filter, 'symmetric') - muA.*muF;
    L = (2*muA.*muF + C1) ./ (muA.^2 + muF.^2 + C1);
    C = (2*sqrt(sigmaA2).*sqrt(sigmaF2) + C2) ./ (sigmaA2 + sigmaF2 + C2);
    S = (sigmaAF + C2/2) ./ (sqrt(sigmaA2).*sqrt(sigmaF2) + C2/2);
    ssim_val = mean2(L .* C .* S);
end

function mi_val = mutual_info_metric(A, B, F)
    A = rgb2gray(im2double(A));
    B = rgb2gray(im2double(B));
    F = rgb2gray(im2double(F));
    miAF = mi_single(A, F);
    miBF = mi_single(B, F);
    mi_val = (miAF + miBF) / 2;
    
    function mi = mi_single(X, Y)
        X = mat2gray(X); Y = mat2gray(Y);
        h = hist3([X(:), Y(:)], "Edges", {linspace(0,1,65), linspace(0,1,65)});
        p = h / sum(h(:));
        px = sum(p, 2); py = sum(p, 1);
        Hx = -sum(px(px>0) .* log2(px(px>0)));
        Hy = -sum(py(py>0) .* log2(py(py>0)));
        Hxy = -sum(p(p>0) .* log2(p(p>0)));
        mi = Hx + Hy - Hxy;
    end
end

function qabf_val = qabf_metric(A, B, F)
    A = rgb2gray(im2double(A));
    B = rgb2gray(im2double(B));
    F = rgb2gray(im2double(F));
    [H, W] = size(A);
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    gA = sqrt(imfilter(A, sobel_x, 'symmetric').^2 + imfilter(A, sobel_y, 'symmetric').^2);
    gB = sqrt(imfilter(B, sobel_x, 'symmetric').^2 + imfilter(B, sobel_y, 'symmetric').^2);
    gF = sqrt(imfilter(F, sobel_x, 'symmetric').^2 + imfilter(F, sobel_y, 'symmetric').^2);
    wA = gA ./ (gA + gB + eps);
    wB = gB ./ (gA + gB + eps);
    QAF = (2 * gA .* gF + eps) ./ (gA.^2 + gF.^2 + eps);
    QBF = (2 * gB .* gF + eps) ./ (gB.^2 + gF.^2 + eps);
    q_map = wA .* QAF + wB .* QBF;
    qabf_val = sum(q_map(:)) / (H * W);
end

function qm_val = qm_metric(A, B, F)
    A = rgb2gray(im2double(A));
    B = rgb2gray(im2double(B));
    F = rgb2gray(im2double(F));
    scores = [];
    for scale = 1:3
        gA = edge(A, 'sobel');
        gB = edge(B, 'sobel');
        gF = edge(F, 'sobel');
        T1 = sum(gF(:) & gA(:)) / (sum(gA(:)) + eps);
        T2 = sum(gF(:) & gB(:)) / (sum(gB(:)) + eps);
        scores = [scores, (T1 + T2) / 2];
        A = imresize(A, 0.5, 'linear');
        B = imresize(B, 0.5, 'linear');
        F = imresize(F, 0.5, 'linear');
    end
    qm_val = mean(scores);
end

function qcb_val = qcb_metric(A, B, F)
    A = rgb2gray(im2double(A));
    B = rgb2gray(im2double(B));
    F = rgb2gray(im2double(F));
    csf_filter = fspecial('gaussian', 5, 1.0);
    A_csf = imfilter(A, csf_filter, 'symmetric');
    B_csf = imfilter(B, csf_filter, 'symmetric');
    F_csf = imfilter(F, csf_filter, 'symmetric');
    muA = mean2(A_csf); muB = mean2(B_csf); muF = mean2(F_csf);
    sigmaA = std2(A_csf); sigmaB = std2(B_csf); sigmaF = std2(F_csf);
    QAF = (4 * muA * muF * sigmaA * sigmaF + eps) ./ ((muA^2 + muF^2) * (sigmaA^2 + sigmaF^2) + eps);
    QBF = (4 * muB * muF * sigmaB * sigmaF + eps) ./ ((muB^2 + muF^2) * (sigmaB^2 + sigmaF^2) + eps);
    qcb_val = (QAF + QBF) / 2;
end

% ==================== 优化1：多特征聚焦度量融合 ====================
% 融合改进拉普拉斯、空间频率、梯度幅度三种特征

% 空间频率聚焦度量（像素级，优化版）
function SF = spatial_frequency(I)
    I = im2double(I);
    % 使用卷积计算局部空间频率（更快速）
    win_size = 7;
    
    % 水平差分核
    dx = [0 0 0; -1 0 1; 0 0 0];
    % 垂直差分核
    dy = [0 -1 0; 0 0 0; 0 1 0];
    
    % 计算差分
    dI_x = imfilter(I, dx, 'symmetric');
    dI_y = imfilter(I, dy, 'symmetric');
    
    % 局部平方和（使用盒式滤波）
    dI_x2 = dI_x.^2;
    dI_y2 = dI_y.^2;
    
    local_x2 = imboxfilt(dI_x2, win_size, 'Padding', 'symmetric') / (win_size^2);
    local_y2 = imboxfilt(dI_y2, win_size, 'Padding', 'symmetric') / (win_size^2);
    
    % 空间频率
    SF = sqrt(local_x2 + local_y2);
    SF = mat2gray(SF);
end

% 梯度幅度聚焦度量（优化版：使用Tenengrad梯度）
% 基于最新研究：Gradient-based MFIF with focus-aware saliency enhancement (2025)
function GM = gradient_magnitude(I)
    I = im2double(I);
    
    % Tenengrad梯度：Sobel滤波器的平方和
    % 这种方法对边缘检测更敏感，更适合聚焦度量
    sobel_x = [-1 0 1; -2 0 2; -1 0 1];
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1];
    Gx = imfilter(I, sobel_x, 'symmetric');
    Gy = imfilter(I, sobel_y, 'symmetric');
    
    % Tenengrad梯度（平方和）
    GM = sqrt(Gx.^2 + Gy.^2);
    
    % 可选：使用局部窗口增强（基于最新研究）
    % 使用盒式滤波计算局部平均梯度
    win_size = 7;
    GM_local = imboxfilt(GM, win_size, 'Padding', 'symmetric') / (win_size^2);
    
    GM = mat2gray(GM_local);
end

% 多尺度空间频率（像素级）
function F_SF = multi_scale_spatial_frequency(I, scale_list)
    I = im2double(I);
    [H, W] = size(I);
    F_SF = zeros(H, W);
    for scale = scale_list
        if scale == 1
            SF_map = spatial_frequency(I);
        else
            % 下采样
            I_scaled = imresize(I, 1/scale, 'linear');
            SF_scaled = spatial_frequency(I_scaled);
            % 上采样回原尺寸
            SF_map = imresize(SF_scaled, [H, W], 'linear');
        end
        F_SF = F_SF + SF_map;
    end
    F_SF = mat2gray(F_SF);
end

% 多特征融合聚焦度量（核心优化）
function FM = multi_feature_focus_measure(I, step_list, w_mml, w_sf, w_gm)
    if nargin < 5
        w_mml = 0.5; w_sf = 0.3; w_gm = 0.2;  % 默认权重
    end
    I = im2double(I);
    
    % 特征1：多尺度修正拉普拉斯
    F_mml = multi_scale_modified_laplacian(I, step_list);
    F_mml = mat2gray(F_mml);
    
    % 特征2：空间频率（多尺度）
    F_sf = multi_scale_spatial_frequency(I, [1, 2]);
    F_sf = mat2gray(F_sf);
    
    % 特征3：梯度幅度
    F_gm = gradient_magnitude(I);
    F_gm = mat2gray(F_gm);
    
    % 加权融合
    FM = w_mml * F_mml + w_sf * F_sf + w_gm * F_gm;
    FM = mat2gray(FM);
end

% ==================== 优化2：引导滤波决策图优化 ====================
% 替代盒式滤波，实现边缘感知的决策图细化
function [FDI1, FDI2] = guided_filter_decision_map(MDI1, MDI2, I_guide, r, eps_val)
    if nargin < 5
        r = 10; eps_val = 0.01;
    end
    MDI1 = im2double(MDI1);
    MDI2 = im2double(MDI2);
    I_guide = im2double(I_guide);
    
    % 引导滤波实现
    [FDI1, FDI2] = guided_filtering(MDI1, MDI2, I_guide, r, eps_val);
end

function [out1, out2] = guided_filtering(I, J, guide, r, eps)
    % I, J: 输入的两张决策图
    % guide: 引导图像
    % r: 窗口半径
    % eps: 正则化参数
    [H, W] = size(guide);
    
    % 引导图均值
    mean_I = imboxfilt(guide, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2;
    % 决策图均值
    mean_J1 = imboxfilt(I, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2;
    mean_J2 = imboxfilt(J, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2;
    
    % 相关量
    corr_IJ1 = imboxfilt(guide .* I, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2;
    corr_IJ2 = imboxfilt(guide .* J, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2;
    
    % 方差
    var_I = imboxfilt(guide.^2, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2 - mean_I.^2;
    
    % 协方差
    cov_J1_I = corr_IJ1 - mean_I .* mean_J1;
    cov_J2_I = corr_IJ2 - mean_I .* mean_J2;
    
    % 线性系数
    a1 = cov_J1_I ./ (var_I + eps);
    b1 = mean_J1 - a1 .* mean_I;
    a2 = cov_J2_I ./ (var_I + eps);
    b2 = mean_J2 - a2 .* mean_I;
    
    % 均值滤波系数
    mean_a1 = imboxfilt(a1, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2;
    mean_b1 = imboxfilt(b1, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2;
    mean_a2 = imboxfilt(a2, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2;
    mean_b2 = imboxfilt(b2, 2*r+1, 'Padding', 'symmetric') / (2*r+1)^2;
    
    % 输出
    out1 = mean_a1 .* guide + mean_b1;
    out2 = mean_a2 .* guide + mean_b2;
    
    % 确保输出在合理范围内
    out1 = max(0, min(1, out1));
    out2 = max(0, min(1, out2));
end

% ==================== 优化3：焦点边界智能处理（改进版） ====================
% 基于最新研究：使用显著性和互补信息进行边界细化
% 检测焦点-散焦边界区域，使用加权插值避免硬切换
function [boundary_mask, weights] = detect_focus_boundary(MDI1, MDI2, threshold)
    if nargin < 3
        threshold = 0.1;
    end
    
    % 计算决策图的差异（改进版）
    diff_map = abs(MDI1 - MDI2);
    
    % 使用形态学梯度检测边界区域（基于最新研究）
    % 形态学梯度可以更好地检测边界
    % se = strel('disk', 1);  % Octave不兼容
    se = ones(3,3);
    grad_MDI1 = imdilate(MDI1, se) - imerode(MDI1, se);
    grad_MDI2 = imdilate(MDI2, se) - imerode(MDI2, se);
    
    % 结合决策图差异和形态学梯度
    boundary_strength = diff_map + 0.5 * (grad_MDI1 + grad_MDI2);
    
    % 差异较大的区域为边界区域（改进阈值检测）
    boundary_mask = (boundary_strength > threshold) & (boundary_strength < 0.9);
    
    % 计算边界区域内的加权权重（改进版）
    weights = zeros(size(MDI1));
    valid_idx = boundary_mask;
    
    if any(valid_idx)
        % 使用平滑过渡权重（避免硬切换）
        weights(valid_idx) = MDI1(valid_idx) ./ (MDI1(valid_idx) + MDI2(valid_idx) + eps);
        
        % 使用高斯滤波平滑权重（基于最新研究）
        weights = imgaussfilt(weights, 1.5);
        
        % 确保权重在合理范围内
        weights = max(0, min(1, weights));
    end
end

%% ===================== 主程序 =====================

% ==================== 图像选择模块 ====================
try
    [filenames, pathname] = uigetfile(...
        {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', '图像文件 (*.jpg, *.png, *.bmp, *.tif)'; ...
         '*.*', '所有文件 (*.*)'}, ...
        '请选择待融合的图像（按住Ctrl或Shift多选）', ...
        'MultiSelect', 'on');
    if isequal(filenames, 0) || isequal(pathname, 0)
        error('用户取消了图像选择操作');
    end
    if ischar(filenames)
        filenames = {filenames};
    end
    num_images = length(filenames);
    if num_images < 2
        error(['请选择至少2张图像进行融合，当前选择了', num2str(num_images), '张']);
    end
    fprintf('成功选择 %d 张图像，开始处理...\n', num_images);
catch ME
    fprintf('错误: %s\n', ME.message);
    return;
end

% ==================== 图像读取与预处理 ====================
fprintf('\n========== 图像读取 ==========\n');
images = cell(1, num_images);
image_names = cell(1, num_images);

first_img_path = fullfile(pathname, filenames{1});
I_ref = imread(first_img_path);
if size(I_ref, 3) == 1
    I_ref = cat(3, I_ref, I_ref, I_ref);
end
[ref_H, ref_W, ~] = size(I_ref);
fprintf('参考图像尺寸: %dx%d\n', ref_H, ref_W);

for i = 1:num_images
    img_path = fullfile(pathname, filenames{i});
    try
        I = imread(img_path);
    catch
        fprintf('警告: 无法读取图像 %s，已跳过\n', filenames{i});
        continue;
    end
    if size(I, 3) == 1
        I = cat(3, I, I, I);
    end
    [H, W, ~] = size(I);
    if H ~= ref_H || W ~= ref_W
        fprintf('图像 %s 尺寸 (%dx%d) 与参考不一致，调整中...\n', filenames{i}, H, W);
        I = imresize(I, [ref_H, ref_W]);
    end
    images{i} = I;
    image_names{i} = filenames{i};
    fprintf('图像%d: %s\n', i, filenames{i});
end

valid_idx = ~cellfun(@isempty, images);
images = images(valid_idx);
image_names = image_names(valid_idx);
num_images = length(images);
if num_images < 2
    error('有效图像不足2张，无法融合');
end

% ==================== 背景优化预处理（输出double [0,1]） ====================
fprintf('\n========== 背景优化预处理 ==========\n');
images_opt = cell(1, num_images);
for i = 1:num_images
    images_opt{i} = background_optimization_core(images{i}, 2);
    fprintf('图像%d 背景优化完成\n', i);
end

% ==================== 图像类型检测（针对小球藻等离散型图像优化） ====================
fprintf('\n========== 图像类型检测 ==========\n');
is_discrete = detect_discrete_image(images_opt{1});
if is_discrete
    fprintf('检测到离散型图像（如小球藻），使用优化参数...\n');
end

% ==================== 迭代融合（增强聚焦记忆 + 清晰区域保护 + 自适应权重） ====================
fprintf('\n========== 分层融合 + 增强聚焦记忆 ==========\n');

% 初始化
fused_result = images_opt{1};
best_focus_map = compute_focus_map(fused_result);

% ==================== 核心可调参数（根据图像类型自适应） ====================
% 离散型图像专用参数（针对小球藻等小对象优化）
if is_discrete
    % 小球藻等离散型图像：使用更小的窗口和分块
    conf_thresh    = 0.55;    % 降低阈值，保护更多清晰区域
    update_delta   = 0.02;    % 更精细的更新阈值
    weak_delta     = 0.005;   % 更精细的弱更新阈值
    win_sizes      = [1, 2, 3];  % 使用更小的窗口捕捉细胞边缘
    ndf_win        = 3;       % 更小的邻距滤波窗口
    bcv_block_size = 6;       % 更小的BCV分块（适应小细胞）
    bcv_nbr_size   = 3;       % 更小的邻域滤波
    guide_r        = 4;       % 更小的引导滤波窗口
    guide_eps      = 0.005;   % 更小的正则化参数
    boundary_thresh = 0.08;  % 更敏感的边界检测
    fprintf('离散图像模式：win=%s, ndf=%d, bcv=%d, guide=%d\n', ...
            mat2str(win_sizes), ndf_win, bcv_block_size, guide_r);
else
    % 通用图像参数
    conf_thresh    = 0.65;
    update_delta   = 0.03;
    weak_delta     = 0.01;
    win_sizes      = [3, 5, 7];
    ndf_win        = 7;
    bcv_block_size = 12;
    bcv_nbr_size   = 7;
    guide_r        = 8;
    guide_eps      = 0.01;
    boundary_thresh = 0.1;
    fprintf('通用图像模式：win=%s, ndf=%d, bcv=%d, guide=%d\n', ...
            mat2str(win_sizes), ndf_win, bcv_block_size, guide_r);
end

% 初始化高置信保护掩码（锁定已有的清晰区域）
conf_protect_mask = get_confidence_mask(best_focus_map, conf_thresh);

% 优化：使用多特征融合聚焦度量生成初始决策图
fprintf('使用优化配置：多特征聚焦度量 + 引导滤波 + 边界智能处理\n');

for i = 2:num_images
    fprintf('第 %d/%d 次融合：融合结果 + 图像%d\n', i-1, num_images-1, i);
    I_curr = fused_result;
    I_new  = images_opt{i};

    % 1. 计算聚焦图 + 双阈值划分更新区域（使用多特征融合）
    new_focus_map = compute_focus_map(I_new);
    strong_update = (new_focus_map > best_focus_map + update_delta);
    weak_update   = (new_focus_map > best_focus_map + weak_delta) & ~strong_update;
    % 总更新区域：排除已锁定的高置信清晰区
    total_update  = (strong_update | weak_update) & ~conf_protect_mask;

    if ~any(total_update(:))
        fprintf('  无有效更新区域，跳过本次融合\n');
        continue;
    end

    % 2. 基于聚焦度生成自适应融合权重
    w_curr_focus = focus2weight(best_focus_map, 0.1, 0.9);
    w_new_focus  = focus2weight(new_focus_map, 0.1, 0.9);

    % 灰度图用于决策图生成
    I_curr_gray = rgb2gray(I_curr);
    I_new_gray  = rgb2gray(I_new);

    % ===== 优化2：使用多特征融合初始决策图 =====
    % 使用多特征聚焦度量替代单纯的MML
    F1 = multi_feature_focus_measure(I_curr_gray, win_sizes, 0.5, 0.3, 0.2);
    F2 = multi_feature_focus_measure(I_new_gray, win_sizes, 0.5, 0.3, 0.2);
    F1 = mat2gray(F1);
    F2 = mat2gray(F2);
    
    % 基于多特征生成初始决策图（互补权重）
    % 使用softmax风格确保权重互补
    IDI1 = F1 ./ (F1 + F2 + eps);
    IDI2 = F2 ./ (F1 + F2 + eps);
    
    % BCV中间决策图（使用自适应分块大小）
    [MDI1, MDI2] = generate_mid_decision_map_bcv(IDI1, IDI2, bcv_block_size, bcv_nbr_size);
    
    % ===== 优化2：使用引导滤波替代盒式滤波进行最终决策图优化 =====
    I_guide = rgb2gray(I_curr);  % 使用当前图像作为引导
    [FDI1, FDI2] = guided_filter_decision_map(MDI1, MDI2, I_guide, guide_r, guide_eps);
    
    % 确保决策图互补（归一化）
    FDI_sum = FDI1 + FDI2 + eps;
    FDI1 = FDI1 ./ FDI_sum;
    FDI2 = FDI2 ./ FDI_sum;
    
    % 决策图扩展为三通道
    F1L = cat(3, FDI1, FDI1, FDI1);
    F2L = cat(3, FDI2, FDI2, FDI2);
    % 高频层决策图（用于细节层）
    F1H = F1L;
    F2H = F2L;

    % 邻距滤波器分层分解（使用向量化优化版本）
    [S_curr, D_curr] = NDF_StructureLayer(I_curr, ndf_win, true);
    [S_new,  D_new]  = NDF_StructureLayer(I_new, ndf_win, true);

    % 3. 分层融合：使用决策图作为权重（简化版，确保权重互补）
    % 结构层融合（S在[0,1]范围）
    fused_S = S_curr .* F1L + S_new .* F2L;
    % 细节层融合（D在[-0.5, 0.5]范围）
    fused_D = D_curr .* F1H + D_new .* F2H;
    % 合并结构层和细节层
    fused_layer = fused_S + fused_D;
    % 归一化到[0,1]范围
    fused_layer = mat2gray(fused_layer);

    % ===== 优化3：焦点边界智能处理（使用自适应阈值） =====
    [boundary_mask, boundary_weights] = detect_focus_boundary(FDI1, FDI2, boundary_thresh);
    if any(boundary_mask(:))
        fprintf('  检测到边界区域（%.2f%%像素），使用智能加权融合\n', 100*mean(boundary_mask(:)));
        % 在边界区域使用加权融合
        bw_3ch = cat(3, boundary_weights, boundary_weights, boundary_weights);
        FDI1_3ch = cat(3, FDI1, FDI1, FDI1);
        FDI2_3ch = cat(3, FDI2, FDI2, FDI2);
        fused_layer = fused_layer .* (1 - bw_3ch) + ...
                      (I_curr .* FDI1_3ch + I_new .* FDI2_3ch) .* bw_3ch;
        fused_layer = mat2gray(fused_layer);
    end

    % 4. 掩码平滑过渡，避免边缘伪影（离散图像使用更小的平滑sigma）
    smooth_sigma = ternaryOp(is_discrete, 1.5, 2.5);
    update_mask_3ch = cat(3, total_update, total_update, total_update);
    update_mask_3ch = imgaussfilt(double(update_mask_3ch), smooth_sigma);
    fused_temp = (1 - update_mask_3ch) .* I_curr + update_mask_3ch .* fused_layer;
    fused_temp = mat2gray(fused_temp);

    % ===== 质量控制：检测融合后清晰区域是否模糊 =====
    % 基于最新研究（Gradient-based MFIF with focus-aware saliency enhancement）
    % 使用Tenengrad梯度检测清晰区域的质量变化
    clear_region_mask = conf_protect_mask;  % 已锁定的清晰区域
    
    % 检查融合质量（梯度值不低于85%阈值）
    [quality_ok, degradation_ratio] = check_fusion_quality(I_curr, fused_temp, clear_region_mask, 0.85);
    
    if ~quality_ok
        % 质量不合格，跳过本次融合，保持原结果
        fprintf('  [质量控制] 融合质量不合格（梯度保留率 %.1f%%），跳过本次融合\n', degradation_ratio*100);
        continue;  % 终止当前迭代，启动下一次
    end
    
    % 质量合格，更新融合结果
    fused_result = fused_temp;

    % 5. 更新全局最优聚焦图 & 刷新保护掩码
    best_focus_map = max(best_focus_map, new_focus_map);
    conf_protect_mask = get_confidence_mask(best_focus_map, conf_thresh);

    update_ratio = 100 * mean(total_update(:));
    fprintf('  融合完成，有效更新像素比例: %.2f%%，质量合格（梯度保留 %.1f%%）\n', update_ratio, degradation_ratio*100);
end

% ==================== 评价指标计算（仅保留原始5项） ====================
fprintf('\n========== 评价指标计算 ==========\n');
ref_A = im2double(images{1});
ref_B = im2double(images{2});

ssim_val  = ssim_metric(fused_result, ref_A);
mi_val    = mutual_info_metric(ref_A, ref_B, fused_result);
qabf_val  = qabf_metric(ref_A, ref_B, fused_result);
qm_val    = qm_metric(ref_A, ref_B, fused_result);
qcb_val   = qcb_metric(ref_A, ref_B, fused_result);

% 仅输出原有5项指标
fprintf('SSIM   = %.4f\n', ssim_val);
fprintf('MI     = %.4f\n', mi_val);
fprintf('QAB/F  = %.4f\n', qabf_val);
fprintf('QM     = %.4f\n', qm_val);
fprintf('QCB    = %.4f\n', qcb_val);

% ==================== 结果展示 ====================
figure('Name', '最终融合结果', 'Position', [100, 100, 800, 600]);
imshow(fused_result);
title(sprintf('MF-BOE 融合结果 (共%d张图像)', num_images), 'FontSize', 12);

figure('Name', '源图像序列', 'Position', [150, 150, 1000, 400]);
num_cols = min(num_images, 5);
num_rows = ceil(num_images / num_cols);
for i = 1:num_images
    subplot(num_rows, num_cols, i);
    imshow(images{i});
    title(sprintf('源图 %d', i), 'FontSize', 10);
end

% 仅绘制原有5项指标柱状图
figure('Name', '评价指标', 'Position', [200, 200, 600, 400]);
names = {'SSIM', 'MI', 'Q^{AB/F}', 'Q_M', 'Q_{CB}'};
vals  = [ssim_val, mi_val, qabf_val, qm_val, qcb_val];

bar(vals, 'FaceColor', [0.2, 0.6, 0.9]);
set(gca, 'XTickLabel', names, 'FontSize', 10);
ylabel('Value'); ylim([0, max(1, max(vals)+0.1)]);
title('融合图像定量评价指标', 'FontSize', 12);
grid on;
for i = 1:5
    text(i, vals(i) + 0.02, sprintf('%.3f', vals(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
end

fprintf('\n========== 融合全部完成 ==========\n');

end
% ==================== Python 调用入口 ====================
function fused_result = image_fusion_algorithm(images_cell)
    fused_result = main_fusion(images_cell);
end
