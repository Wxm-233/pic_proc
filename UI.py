#-*-coding:UTF-8 -*-
from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np


def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业一: 色彩处理工具') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')
                H = gr.Slider(minimum=-1, maximum=1, value=0, step=0.01, label='色相(H)', show_label=True)
                L = gr.Slider(minimum=-1, maximum=1, value=0, step=0.01, label='亮度(L)', show_label=True)
                S = gr.Slider(minimum=-1, maximum=1, value=0, step=0.01, label='饱和度(S)', show_label=True)
            with gr.Column():
                output_image = gr.Image(type='numpy', label='标准输出图像', interactive=False)
                my_output_image = gr.Image(type='numpy', label='自制输出图像', interactive=False)
        with gr.Row():
            run_button = gr.Button(value='运行')

        run_button.click(fn=process,
                        inputs=[input_image, H, L, S],
                        outputs=[output_image, my_output_image])
    return demo

def create_demo_hw2(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业二: 大小调整工具') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')  
                size_x = gr.Slider(minimum=0.5, maximum=4, value=1, step=0.1, label='缩放倍数(横向)', show_label=True)
                size_y = gr.Slider(minimum=0.5, maximum=4, value=1, step=0.1, label='缩放倍数(纵向)', show_label=True)
                mode = gr.Radio(['最近邻插值', '双线性插值', '双三次插值', 'Lanczos插值'], label='插值方式', value='双线性插值')
                rotate_angle = gr.Radio(['0', '90', '180', '270'], label='旋转角度(顺时针)', value='0')
                biascut_percent = gr.Slider(minimum=0, maximum=1, value=0, step=0.01, label='斜切百分比', show_label=True)
            with gr.Column():
                output_image = gr.Image(type='numpy', label='标准输出图像', interactive=False)
                output_image2 = gr.Image(type='numpy', label='手动实现输出', interactive=False)
                run_button = gr.Button(value='运行')
                diff_image = gr.Image(type='numpy', label='两图像之差', interactive=False)

        run_button.click(fn=process,
                        inputs=[input_image, size_x, size_y, mode, rotate_angle, biascut_percent],
                        outputs=[output_image, output_image2, diff_image])
    return demo


def create_demo_hw3(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业三: 图像生成工具') 
        with gr.Row():
            with gr.Column():
                # 通过输入框输入seed（范围为0-INT_MAX）
                seed = gr.Number(label='种子（0则随机生成）', minimum=0, maximum=2147483647, value=0, step=1)
                # 选择性别
                gender = gr.Radio(['男','女','无'], label='性别倾向', value='无')
            with gr.Column():
                output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
                run_button = gr.Button(value='运行')

        run_button.click(fn=process,
                        inputs=[seed, gender],
                        outputs=[output_image])
    return demo

def create_demo_hw4(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业四: 图像去噪工具') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')
                mode = gr.Radio(['双边滤波', 'NLM滤波', '导向滤波', '基于盒式滤波优化的快速导向滤波', '手动双边滤波'], label='滤波方式', value='双边滤波')
            with gr.Column():
                output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
                run_button = gr.Button(value='运行')

        run_button.click(fn=process,
                        inputs=[input_image, mode],
                        outputs=[output_image])
    return demo

def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown('## 作业五: XXX工具') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='输入图像')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='输出图像', interactive=False)
                run_button = gr.Button(value='运行')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo