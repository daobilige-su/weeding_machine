import time, sys, os
from cv_bridge import CvBridge
import cv2
import numpy as np

#
def add_rectangle_to_video(input_video_path, output_video_path):
    video_path = input_video_path
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Failed to open video.")
        exit()

    # 获取视频的帧率、宽度和高度
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建一个空白的透明图像（即黑色背景）
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # 在透明图像上绘制一个浅绿色的矩形mask
    # 这里假设要在视频的 (100, 100) 到 (300, 300) 区域添加浅绿色的矩形mask
    cv2.rectangle(mask, (650, 290), (700, 330), (127, 255, 0), -1)  # (B, G, R) = (127, 255, 0)，-1 表示填充矩形

    # 设置保存处理后视频的参数
    output_video_path = output_video_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 循环处理视频的每一帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将浅绿色的矩形mask叠加到当前帧上
        masked_frame = np.where(mask == 0, frame, mask)

        # 写入处理后的帧到输出视频文件
        out.write(masked_frame)

        # 显示处理后的帧（可选）
        cv2.imshow('Video with Mask', masked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved as {output_video_path}")
if __name__ == "__main__":
    if len( sys.argv ) ==7 :
        input_video = '/home/zhong/my_workspace/src/weeding_machine/yolov5/data/1.mp4 '  # Replace with your input video file path
        output_video = '/home/zhong/my_workspace/src/weeding_machine/yolov5/data/1_reg.mp4'  # Replace with desired output video path
        frame_number = 100  # Frame number where you want to add the rectangle (adjust as needed)
        x = 300  # x-coordinate of the center of the rectangle
        y = 200  # y-coordinate of the center of the rectangle
        size = 100  # Size of the square
        color = (152, 251, 152)  # RGB color of the rectangle (light gray in this case)
        thickness = 2  # Thickness of the rectangle border

        # add_rectangle_to_video(input_video, output_video, frame_number, x, y, size, color, thickness)
        add_rectangle_to_video(*sys.argv[1:3])
    else:
        print( "Usage: video2bag videofilename bagfilename")

# python add_reg_in_video.py /home/zhong/my_workspace/src/weeding_machine/yolov5/data/3.mp4 /home/zhong/my_workspace/src/weeding_machine/yolov5/data/3_reg.mp4  100 300 200 50
