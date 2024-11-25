import numpy as np
import math
import cv2 as cv
import gradio as gr

# hw1

def HSL_adjust(input_image: gr.Image, pH, pL, pS):
    output_image = np.zeros_like(input_image)
    H = input_image[..., 0]
    L = input_image[..., 1]
    S = input_image[..., 2]
    output_image[..., 0] = H * (1+pH) if pH<0 else H + (180 - H) * pH
    output_image[..., 1] = L * (1+pL) if pL<0 else L + (255 - L) * pL
    output_image[..., 2] = S * (1+pS) if pS<0 else S + (255 - S) * pS
    return output_image

def my_cvtColor_RGB2HLS(input_image):
    R = input_image[..., 0].astype(np.float32)
    G = input_image[..., 1].astype(np.float32)
    B = input_image[..., 2].astype(np.float32)
    
    inter = R*R + G*G + B*B - R*G - R*B - B*G
    theta = np.where(inter==0, 0, np.arccos((2*R - G - B)/(2 * np.sqrt(inter))))
    theta = np.rad2deg(theta)

    H = np.where(B<=G, theta, 360-theta) / 2
    S = np.where(R+G+B == 0, 0, 1 - 3 * np.minimum(np.minimum(R, G), B) / (R + G + B)) * 255
    L = (R + G + B) / 3

    hls_image = np.zeros_like(input_image)
    hls_image[..., 0] = H
    hls_image[..., 1] = L
    hls_image[..., 2] = S

    return hls_image

def my_cvtColor_HLS2RGB(input_image):
    H = input_image[..., 0].astype(np.float32)
    L = input_image[..., 1].astype(np.float32)
    S = input_image[..., 2].astype(np.float32)

    H *= 2
    H = np.deg2rad(H)
    S = S / 255

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            h = H[i, j]
            l = L[i, j]
            s = S[i, j]
            if h >= 0 and h < 2*np.pi/3:
                r = l * (1 + s * np.cos(h) / np.cos(np.pi/3 - h))
                b = l * (1 - s)
                g = 3 * l - r - b
            elif h >= 2*np.pi/3 and h < 4*np.pi/3:
                h -= 2*np.pi/3
                r = l * (1 - s)
                g = l * (1 + s * np.cos(h) / np.cos(np.pi/3 - h))
                b = 3 * l - r - g
            else:
                h -= 4*np.pi/3
                g = l * (1 - s)
                b = l * (1 + s * np.cos(h) / np.cos(np.pi/3 - h))
                r = 3 * l - g - b

            R[i, j] = r
            G[i, j] = g
            B[i, j] = b

    # R = np.where(np.logical_and(H >= 0, H < 2*np.pi/3), L * (1 + S * np.cos(H) / np.cos(np.pi/3 - H)), 0)
    # B = np.where(np.logical_and(H >= 0, H < 2*np.pi/3), L * (1 - S), 0)
    # G = np.where(np.logical_and(H >= 0, H < 2*np.pi/3), 3 * L - R - B, 0) 

    # H2 = np.where(np.logical_and(H >= 2*np.pi/3, H < 4*np.pi/3), H - 2*np.pi/3, 0)
    # R = np.where(np.logical_and(H >= 2*np.pi/3, H < 4*np.pi/3), L * (1 - S), R)
    # G = np.where(np.logical_and(H >= 2*np.pi/3, H < 4*np.pi/3), L * (1 + S * np.cos(H2) / np.cos(np.pi/3 - H2)), G)
    # B = np.where(np.logical_and(H >= 2*np.pi/3, H < 4*np.pi/3), 3 * L - R - G, B)

    # H2 = np.where(H >= 4*np.pi/3, H - 4*np.pi/3, 0)
    # G = np.where(H >= 4*np.pi/3, L * (1 - S), G)
    # B = np.where(H >= 4*np.pi/3, L * (1 + S * np.cos(H2) / np.cos(np.pi/3 - H2)), B)
    # R = np.where(H >= 4*np.pi/3, 3 * L - G - B, R)


    # if H >= 0 and H < 2*np.pi/3:
    #     R = L * (1 + S * np.cos(H) / np.cos(np.pi/3 - H))
    #     B = L * (1 - S)
    #     G = 3 * L - R - B
    # elif H >= 2*np.pi/3 and H < 4*np.pi/3:
    #     H -= 2*np.pi/3
    #     R = L * (1 - S)
    #     G = L * (1 + S * np.cos(H) / np.cos(np.pi/3 - H))
    #     B = 3 * L - R - G
    # else:
    #     H -= 4*np.pi/3
    #     G = L * (1 - S)
    #     B = L * (1 + S * np.cos(H) / np.cos(np.pi/3 - H))
    #     R = 3 * L - G - B

    output_image = np.zeros_like(input_image)
    output_image[..., 0] = R.astype(np.uint8)
    output_image[..., 1] = G.astype(np.uint8)
    output_image[..., 2] = B.astype(np.uint8)
    return output_image

# hw2

def my_diff_image(image1, image2):
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape
    height = min(height1, height2)
    width = min(width1, width2)
    diff_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            diff_image[i, j] = image1[i, j] - image2[i, j]
    return diff_image

def my_resize(input_image, size_x, size_y, mode):
    match mode:
        case cv.INTER_NEAREST:
            return my_inter_nearest(input_image, size_x, size_y)
        case cv.INTER_LINEAR:
            return my_inter_linear(input_image, size_x, size_y)
        case cv.INTER_CUBIC:
            return my_inter_cubic(input_image, size_x, size_y)
        case cv.INTER_LANCZOS4:
            return my_inter_lanczos(input_image, size_x, size_y)
        case _:
            return my_inter_linear(input_image, size_x, size_y)

def my_inter_nearest(input_image, size_x, size_y):
    height, width, _ = input_image.shape
    n_height = int(height * size_y)
    n_width = int(width * size_x)
    n_image = np.zeros((n_height, n_width, 3), dtype=np.uint8)
    for i in range(n_height):
        for j in range(n_width):
            x = min(round(i / size_y), height-1)
            y = min(round(j / size_x), width-1)
            n_image[i, j] = input_image[x, y]
    return n_image

def my_inter_linear(input_image, size_x, size_y):
    height, width, _ = input_image.shape
    n_height = int(height * size_y)
    n_width = int(width * size_x)
    n_image = np.zeros((n_height, n_width, 3), dtype=np.uint8)
    for i in range(n_height):
        for j in range(n_width):
            x = i / size_y
            y = j / size_x
            x1 = math.floor(x)
            x2 = min(x1+1, height-1)
            y1 = math.floor(y)
            y2 = min(y1+1, width-1)
            c1 = (x2 - x) * (y2 - y)
            c2 = (x - x1) * (y2 - y)
            c3 = (x2 - x) * (y - y1)
            c4 = (x - x1) * (y - y1)
            n_image[i, j] = c1 * input_image[x1, y1] + c2 * input_image[x2, y1] + c3 * input_image[x1, y2] + c4 * input_image[x2, y2]
    return n_image

def my_inter_cubic(input_image, size_x, size_y):
    def W(x, a = -1)->float:
        x = abs(x)
        if x <= 1:
            return (a+2) * pow(x,3) - (a+3) * pow(x, 2) + 1
        elif x > 1 and x < 2:
            return a * pow(x,3) - 5*a * pow(x,2) + 8*a * x - 4*a
        else: return 0
    height, width, _ = input_image.shape
    n_height = int(height * size_y)
    n_width = int(width * size_x)
    n_image = np.zeros((n_height, n_width, 3), dtype=np.uint8)
    for i in range(n_height):
        for j in range(n_width):
            x = i / size_y
            y = j / size_x
            pivot_x = math.floor(x)
            pivot_y = math.floor(y)
            temp: float = 0
            for m in range(-1, 3):
                for n in range(-1, 3):
                    xi = pivot_x + m
                    yj = pivot_y + n
                    if xi < 0 or xi >= height or yj < 0 or yj >= width:
                        continue
                    temp += input_image[xi, yj] * W(x-xi) * W(y-yj)
            n_image[i, j] = np.clip(temp, 0, 255).astype(np.uint8)
    return n_image

def my_inter_lanczos(input_image, size_x, size_y):
    def Sinc(x)->float:
        if x == 0:
            return 1
        return math.sin(math.pi * x) / (math.pi * x)
    def Lanczos(x, a)->float:
        if x == 0:
            return 1
        if abs(x) >= a:
            return 0
        return Sinc(x) * Sinc(x/a)
    a = 2 if size_x * size_y <= 1 else 3
    height, width, _ = input_image.shape
    n_height = int(height * size_y)
    n_width = int(width * size_x)
    n_image = np.zeros((n_height, n_width, 3), dtype=np.uint8)
    for i in range(n_height):
        for j in range(n_width):
            x = i / size_y
            y = j / size_x
            pivot_x = math.floor(x)
            pivot_y = math.floor(y)
            temp: float = 0
            for m in range(-a+1, a+1):
                for n in range(-a, a+1):
                    xi = pivot_x + m
                    yj = pivot_y + n
                    if xi < 0 or xi >= height or yj < 0 or yj >= width:
                        continue
                    temp = temp + Lanczos(x-xi, a) * Lanczos(y-yj, a) * input_image[xi, yj]  
            n_image[i, j] = np.clip(temp, 0, 255).astype(np.uint8)
    return n_image

# def my_rotate(input_image, angle):
#     angle = math.radians(angle) % (2 * math.pi)
#     height, width, _ = input_image.shape
#     if angle == 0:
#         return input_image
#     if (0 < angle <= math.pi/2):
#         right_shift = height * math.sin(angle)
#         up_shift = 0
#     elif (math.pi/2 < angle <= math.pi):
#         right_shift = height * math.sin(angle) - width * math.cos(angle)
#         up_shift = -height * math.cos(angle)
#     elif (math.pi < angle <= 3*math.pi/2):
#         right_shift = -height * math.sin(angle) - width * math.cos(angle)
#         up_shift = -width * math.sin(angle)
#     elif (3*math.pi/2 < angle < 2*math.pi):
#         right_shift = 0
#         up_shift = -width * math.sin(angle) + height * math.cos(angle)

#     M1 = np.float32([[1, 0, right_shift], [0, 1, -up_shift]])

#     output_image = my_warp_affine(input_image, M1, (height, width))

#     M2 = np.float32([[math.cos(angle), math.sin(angle), right_shift], [-math.sin(angle), math.cos(angle), -up_shift]])
#     # 计算旋转后的图像大小
#     n_height = int(width * abs(math.sin(angle)) + height * abs(math.cos(angle)))
#     n_width = int(height * abs(math.sin(angle)) + width * abs(math.cos(angle)))
    
#     # 计算位移量
#     return my_warp_affine(output_image, M2, (n_height, n_width))

def my_warp_affine(input_image, M, dsize):
    i_height, i_width, _ = input_image.shape
    o_height, o_width = dsize
    n_image = np.zeros((o_height, o_width, 3), dtype=np.uint8)
    M = cv.invertAffineTransform(M)
    for o_x in range(o_width):
        for o_y in range(o_height):
            i_x = int(M[0, 0] * o_x + M[0, 1] * o_y + M[0, 2])
            i_y = int(M[1, 0] * o_x + M[1, 1] * o_y + M[1, 2])
            if i_x < 0 or i_x >= i_width or i_y < 0 or i_y >= i_height:
                continue
            n_image[o_y, o_x] = input_image[i_y, i_x]
    return n_image

def my_guided_filter(src, guide, r, eps): # 只输入一张图像
    h, w = src.shape[:2]
    q = np.zeros_like(src, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            # 定义窗口范围
            i_min, i_max = max(i-r, 0), min(h, i+r+1)
            j_min, j_max = max(j-r, 0), min(w, j+r+1)

            # 提取局部窗口
            I_win = src[i_min:i_max, j_min:j_max]
            p_win = guide[i_min:i_max, j_min:j_max]

            # 计算局部统计量
            mean_I = np.mean(I_win)
            mean_p = np.mean(p_win)
            mean_Ip = np.mean(I_win * p_win)
            cov_Ip = mean_Ip - mean_I * mean_p
            var_I = np.mean(I_win * I_win) - mean_I * mean_I

            # 计算线性系数 a 和 b
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I

            # 输出q
            q[i, j] = a * src[i, j] + b
    
    return q

def my_fast_guided_filter(I, p, r, eps):
    # 盒式滤波
    mean_I = cv.boxFilter(I, -1, (r, r))
    mean_p = cv.boxFilter(p, -1, (r, r))

    corr_I = cv.boxFilter(I*I, -1, (r, r))
    corr_Ip = cv.boxFilter(I*p, -1, (r, r))

    # 计算协方差
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    # 计算a, b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 对a, b盒式滤波，并输出q
    mean_a = cv.boxFilter(a, -1, (r, r))
    mean_b = cv.boxFilter(b, -1, (r, r))
    q = mean_a * I + mean_b
    output_image = np.clip(q, 0, 255).astype(np.uint8)
    return output_image

def my_bilateral_filter(input_image, d, sigma_color, sigma_space):
    def G(x, sigma):
        return np.exp(-x**2 / (2 * sigma**2))

    height, width = input_image.shape[:2]
    n_image = np.zeros_like(input_image, dtype=np.uint8)
    r = d // 2
    for i in range(height):
        for j in range(width):
            W_p = 0
            I_p = 0
            for m in range(-r, d-r):
                for n in range(-r, d-r):
                    n_i, n_j = i + m, j + n
                    if n_i < 0 or n_i >= height or n_j < 0 or n_j >= width:
                        continue
                    range_weight = G(np.mean(input_image[i, j]) - np.mean(input_image[n_i, n_j]), sigma_color)
                    spatial_weight = G(math.sqrt(m**2 + n**2), sigma_space)
                    W_p += range_weight * spatial_weight
                    I_p += range_weight * spatial_weight * input_image[n_i, n_j]
            n_image[i, j] = np.clip(I_p / W_p, 0, 255)
    return n_image