#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[infer_onnx_cpu.py]
- 사전에 export해둔 ONNX 모델을 ONNX Runtime(CPU)으로 추론 검증.
- 입력은 1채널 IR(16-bit RAW 또는 8-bit PNG) 가정.
- 전처리/후처리 위치를 주석으로 명확히 표시.
"""

import argparse
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path

def load_ir_image_as_1x1hw(image_path: str, expect_h: int, expect_w: int):
    """
    IR 입력 이미지를 (1,1,H,W) float32 텐서로 변환
    - 16비트 RAW/PNG → 0~65535 범위를 0~1 float로 정규화 (필요시 변경)
    - 8비트라면 0~255로 읽혀옴 → 0~1 스케일로 변경
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 8/16bit 자동
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {image_path}")

    # 단일 채널만 허용
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 크기 강제(모델 고정 입력과 동일해야 함)
    if img.shape[0] != expect_h or img.shape[1] != expect_w:
        img = cv2.resize(img, (expect_w, expect_h), interpolation=cv2.INTER_NEAREST)

    # dtype별 정규화
    if img.dtype == np.uint16:
        x = img.astype(np.float32) / 65535.0
    else:
        x = img.astype(np.float32) / 255.0

    # (H,W) -> (1,1,H,W)
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0).astype(np.float32)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="ONNX 모델 경로")
    ap.add_argument("--input", required=True, help="입력 이미지 경로(8/16bit 단일 채널)")
    ap.add_argument("--shape", default="1,1,480,640", help="N,C,H,W (모델 고정 입력)")
    ap.add_argument("--out", default="pred.png", help="출력 저장 경로(시각화용)")
    args = ap.parse_args()

    # 입력 모양 파싱
    N, C, H, W = [int(v) for v in args.shape.split(",")]
    assert N == 1 and C == 1, "본 예제는 1x1xHxW만 가정"

    # 세션 로드(CPU)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # 전처리
    x = load_ir_image_as_1x1hw(args.input, H, W)

    # 추론
    y = sess.run([out_name], {inp_name: x})[0]  # [1,1,H,W] 가정
    y = np.squeeze(y, axis=(0,1))               # (H,W)

    # 후처리(시각화용 0~1 -> 0~255)
    y_vis = np.clip(y, 0.0, 1.0) * 255.0
    y_vis = y_vis.astype(np.uint8)
    cv2.imwrite(args.out, y_vis)
    print(f"[OK] 추론 완료. 시각화 저장: {args.out}")

if __name__ == "__main__":
    main()
