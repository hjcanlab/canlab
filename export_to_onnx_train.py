#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[목적]
- 사용자가 제공한 train.py 문맥(특히 get_net(...)로 생성한 'skip' 네트워크)을 기준으로,
  PyTorch 체크포인트(.pth)를 ONNX(.onnx)로 내보내는 스크립트.
- '전체 모델 저장(torch.save(model))'과 'state_dict만 저장(torch.save(model.state_dict()))'를
  모두 지원하도록 자동 감지 로직을 포함.

[중요 전제]
- TIDL을 쓸 것이므로 '정적 입력 크기'로 내보내는 것을 기본으로 함.
- 입력은 1채널(IR)이라고 가정(train.py에서 n_channels=1, input_depth=1).
  필요 시 --input_shape 인자로 변경 가능.

[사용 예시]
python export_to_onnx_from_train.py \
  --ckpt ./results/train/20250819/model_full_20250819_4900.pth \
  --onnx_out /workspace/models/thermal_skip_480x640.onnx \
  --input_shape 1,1,480,640 \
  --config_json ./results/train/20250819/model_config_20250819.json \
  --opset 12
"""

import os
import sys
import json
import argparse
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnx import checker
import onnxruntime as ort

# ----------------------------
# (1) train.py에서 사용한 네트 생성 함수/모듈 불러오기
#     - 같은 프로젝트 루트에서 실행한다고 가정
#     - get_net(...) 시그니처는 train.py와 동일하게 호출
# ----------------------------
try:
    from models import get_net
except Exception as e:
    print("[오류] models.get_net import 실패. PYTHONPATH 또는 현재 실행 디렉터리를 확인하세요.")
    raise

# (선택) onnx-simplifier 사용 가능하면 켜기
try:
    from onnxsim import simplify as onnx_simplify
    HAS_SIMPLIFIER = True
except Exception:
    HAS_SIMPLIFIER = False


def parse_shape(shape_str: str) -> Tuple[int, int, int, int]:
    """
    입력 문자열 "N,C,H,W" -> (N,C,H,W) 정수 튜플로 변환
    예: "1,1,480,640" -> (1,1,480,640)
    """
    parts = [s.strip() for s in shape_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"입력 shape 포맷 오류: {shape_str} (예: 1,1,480,640)")
    N, C, H, W = [int(x) for x in parts]
    return (N, C, H, W)


def build_net_from_config(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """
    JSON 또는 기본값으로부터 train.py의 get_net(...)을 그대로 호출하여 네트워크 생성.
    - train.py에 있던 기본 파라미터를 그대로 사용(사용자 JSON이 있으면 그것을 우선)
    - 출력/입력 채널(1채널 IR)을 반영
    """
    # train.py의 기본값(사용자 코드 기반)
    default_cfg = {
        "input_depth": 1,
        "NET_TYPE": "skip",
        "pad": "reflection",
        "upsample_mode": "bilinear",
        "n_channels": 1,
        "skip_n33d": 128,
        "skip_n33u": 128,
        "skip_n11": 4,
        "num_scales": 5,
        "downsample_mode": "stride"
    }
    # 사용자 cfg가 있으면 덮어씀
    default_cfg.update(cfg or {})

    # get_net(...) 호출
    net = get_net(
        input_depth       = default_cfg["input_depth"],
        NET_TYPE          = default_cfg["NET_TYPE"],
        pad               = default_cfg["pad"],
        upsample_mode     = default_cfg["upsample_mode"],
        n_channels        = default_cfg["n_channels"],
        skip_n33d         = default_cfg["skip_n33d"],
        skip_n33u         = default_cfg["skip_n33u"],
        skip_n11          = default_cfg["skip_n11"],
        num_scales        = default_cfg["num_scales"],
        downsample_mode   = default_cfg["downsample_mode"],
    )
    net = net.to(device)
    net.eval()
    return net


def load_checkpoint(ckpt_path: str):

    """
    체크포인트를 로드하고, 아래 중 무엇인지 판별:
    1) 전체 모델(torch.save(model))  -> nn.Module 객체로 로드됨
    2) state_dict(torch.save(model.state_dict())) -> dict(key->Tensor)
    3) dict 안에 다시 'state_dict' 키를 가진 케이스 -> dict['state_dict'] 사용

    반환:
    - obj: nn.Module 이거나, state_dict(dict)
    - is_full_model: True/False
    """
    
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"체크포인트가 존재하지 않습니다: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")

    # 1) 전체 모델 저장 형태인지 확인
    if isinstance(obj, nn.Module):
        return obj, True

    # 2) dict 형태인 경우
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"], False
        else:
            # 그냥 state_dict일 가능성
            # 키의 값들이 Tensor인지 대략 체크
            tensor_like = all(hasattr(v, "shape") for v in obj.values())
            if tensor_like:
                return obj, False

    # 그 외는 지원하지 않음
    raise RuntimeError("체크포인트 형식을 인식하지 못했습니다. (전체 모델 or state_dict 형태인지 확인)")



def export_onnx(
    model: nn.Module,
    onnx_out: str,
    input_shape: Tuple[int, int, int, int],
    input_name: str = "input",
    output_names = ("output",),
    opset: int = 12,
    simplify: bool = False,
    verify_run: bool = True,
):
    """
    ONNX 내보내기 + (선택) 단순화 + (선택) onnxruntime로 1회 실행 검증
    - TIDL을 고려하여 dynamic_axes는 사용하지 않음(정적 shape).
    - bilinear upsample, reflection pad 등은 opset 12에서 일반적으로 변환 가능.
      만약 변환 에러가 나면 upsample 모드나 align_corners 등의 파라미터를 점검.
    """
    device = next(model.parameters()).device
    N, C, H, W = input_shape

    # 더미 입력 생성 (정적 크기)
    dummy = torch.randn((N, C, H, W), dtype=torch.float32, device=device)

    # 내보내기
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_out,
            input_names=[input_name],
            output_names=list(output_names),
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,                # 정적 shape 강제
            keep_initializers_as_inputs=False
        )

    print(f"[OK] ONNX 저장: {onnx_out}")

    # onnx.checker
    m = onnx.load(onnx_out)
    checker.check_model(m)
    print("[OK] onnx.checker 통과")

    # (선택) 단순화
    if simplify and HAS_SIMPLIFIER:
        print("[정보] onnx-simplifier로 그래프 단순화 시도...")
        simplified_model, ok = onnx_simplify(m)
        if ok:
            onnx.save(simplified_model, onnx_out)
            print("[OK] 단순화 완료(덮어씀):", onnx_out)
        else:
            print("[경고] 단순화 실패. 원본 모델 유지.")
    elif simplify and not HAS_SIMPLIFIER:
        print("[경고] onnxsim 모듈 없음. 단순화 건너뜀.")

    # (선택) onnxruntime로 1회 실행 검증(CPU EP)
    if verify_run:
        sess = ort.InferenceSession(onnx_out, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy_np = np.random.randn(N, C, H, W).astype(np.float32)
        _ = sess.run(None, {inp_name: dummy_np})
        print("[OK] onnxruntime 실행 검증 성공")


def main():
    parser = argparse.ArgumentParser(description="train.py 맥락의 모델을 ONNX로 내보내기")
    parser.add_argument("--ckpt", type=str, required=True, help="체크포인트 경로(.pth)")
    parser.add_argument("--onnx_out", type=str, required=True, help="출력 ONNX 경로")
    parser.add_argument("--input_shape", type=str, default="1,1,480,640",
                        help="정적 입력 크기 'N,C,H,W' (기본: 1,1,480,640)")
    parser.add_argument("--config_json", type=str, default="",
                        help="(선택) train.py에서 저장한 model_config_*.json 경로")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 버전(기본=12)")
    parser.add_argument("--simplify", action="store_true", help="onnx-simplifier 사용")
    args = parser.parse_args()

    # 0) 장치: GPU 불필요. CPU로 고정(ONNX export/검증 용도)
    device = torch.device("cpu")

    # 1) 입력 크기 파싱
    input_shape = parse_shape(args.input_shape)  # (N,C,H,W)

    # 2) (선택) 모델 구성 JSON 로드
    user_cfg = {}
    if args.config_json:
        if not os.path.isfile(args.config_json):
            raise FileNotFoundError(f"config_json 파일을 찾을 수 없습니다: {args.config_json}")
        with open(args.config_json, "r") as f:
            user_cfg = json.load(f)

    # 3) 체크포인트 로드 (전체 모델인지, state_dict인지 자동 감지)
    obj, is_full_model = load_checkpoint(args.ckpt)

    # 4) 모델 구성
    if is_full_model:
        # 4-1) 전체 모델이 이미 들어있음 -> 장치 이동 + eval
        model = obj.to(device)
        model.eval()
        print("[정보] 전체 모델 체크포인트를 로드했습니다.")
    else:
        # 4-2) state_dict만 있는 경우 -> 동일 구조 생성 후 주입
        print("[정보] state_dict 체크포인트를 로드했습니다. 동일 구조의 네트 생성 중...")
        model = build_net_from_config(user_cfg, device=device)
        missing, unexpected = model.load_state_dict(obj, strict=False)
        if missing:
            print("[경고] 누락된 키:", missing)
        if unexpected:
            print("[경고] 예상치 못한 키:", unexpected)
        model.eval()

    # 5) ONNX Export
    onnx_out_dir = os.path.dirname(args.onnx_out)
    if onnx_out_dir:
        os.makedirs(onnx_out_dir, exist_ok=True)

    export_onnx(
        model=model,
        onnx_out=args.onnx_out,
        input_shape=input_shape,
        input_name="input",
        output_names=("output",),  # 필요 시 사용자 모델에 맞게 이름 늘리기
        opset=args.opset,
        simplify=args.simplify,
        verify_run=True,
    )

    print("\n[완료] ONNX Export 성공:", args.onnx_out)
    print("[다음 단계] 이제 이 ONNX를 가지고 TIDL 컴파일(-c) 하시면 됩니다.")


if __name__ == "__main__":
    main()
