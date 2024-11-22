#-*-coding:UTF-8 -*-
import numpy as np
import gradio as gr
import cv2 as cv
import math as math

from helper_functions import *
from DCGAN import *

def function_hw1(input_image: gr.Image, H, L, S):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    hls_image = cv.cvtColor(input_image, cv.COLOR_RGB2HLS)
    hls_image = HSL_adjust(hls_image, H, L, S)
    output_image = cv.cvtColor(hls_image, cv.COLOR_HLS2RGB)

    my_hls_image = my_cvtColor_RGB2HLS(input_image)
    # my_hls_image = cv.cvtColor(input_image, cv.COLOR_RGB2HLS)
    my_hls_image = HSL_adjust(my_hls_image, H, L, S)
    my_output_image = my_cvtColor_HLS2RGB(my_hls_image)
    # my_output_image = cv.cvtColor(my_hls_image, cv.COLOR_HLS2RGB)
    return output_image, my_output_image

def function_hw2(input_image, size_x, size_y, mode, rotate_angle, biascut_percent):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    match mode:
        case '最近邻插值':
            mode = cv.INTER_NEAREST
        case '双线性插值':
            mode = cv.INTER_LINEAR
        case '双三次插值':
            mode = cv.INTER_CUBIC
        case 'Lanczos插值':
            mode = cv.INTER_LANCZOS4
        case _:
            mode = cv.INTER_LINEAR

    # resize
    output_image = cv.resize(input_image, None, fx=size_x, fy=size_y, interpolation=mode)
    output_image2 = my_resize(input_image, size_x, size_y, mode)

    # rotate
    match rotate_angle:
        case '0':
            pass
        case '90':
            output_image = cv.rotate(output_image, cv.ROTATE_90_CLOCKWISE)
            output_image2 = cv.rotate(output_image2, cv.ROTATE_90_CLOCKWISE)
        case '180':
            output_image = cv.rotate(output_image, cv.ROTATE_180)
            output_image2 = cv.rotate(output_image2, cv.ROTATE_180)
        case '270':
            output_image = cv.rotate(output_image, cv.ROTATE_90_COUNTERCLOCKWISE)
            output_image2 = cv.rotate(output_image2, cv.ROTATE_90_COUNTERCLOCKWISE)
        case _:
            pass
    # biascut
    output_image = cv.warpAffine(output_image, np.float32([[1, biascut_percent, 0], [0, 1, 0]]), (output_image.shape[1], output_image.shape[0]))
    output_image2 = my_warp_affine(output_image2, np.float32([[1, biascut_percent, 0], [0, 1, 0]]), (output_image2.shape[0], output_image2.shape[1]))
    diff_image = my_diff_image(output_image, output_image2)
    return output_image, output_image2, diff_image

def function_hw3(seed, gender):
    if seed is None:
        raise gr.Error('内部错误', duration=5)
    if seed == 0:
        seed = np.random.randint(0, 2147483647)
    output_image = generate_image(seed, gender)
    return output_image

def function_hw4(input_image, mode):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    match mode:
        case '双边滤波':
            output_image = cv.bilateralFilter(input_image, 9, 75, 75)
        case 'NLM滤波':
            output_image = cv.fastNlMeansDenoisingColored(input_image, None, 10, 10, 7, 21)
        case '导向滤波':
            output_image = my_guided_filter(input_image, input_image, 9, 0.1)
        case '基于盒式滤波优化的快速导向滤波':
            output_image = my_fast_guided_filter(input_image, input_image, 9, 0.1)
        case '手动双边滤波':
            output_image = my_bilateral_filter(input_image, 9, 75, 75)
        case _:
            output_image = input_image
    return output_image

def function_hw5(input_image):
    if input_image is None:
        raise gr.Error('输入错误：在处理之前请先输入一张图像', duration=5)
    output_image = input_image
    # 请补充作业5的图像处理代码
    return output_image