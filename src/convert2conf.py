import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor

# 1. config.py와 pth 경로
config_file = 'config.py'
checkpoint_file = 'iter_80000.pth'

# 2. config 로드
cfg = Config.fromfile(config_file)

# 3. 모델 수동 빌드 (init_segmentor 대신)
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
model.CLASSES = ['_background', 'building']  # 너의 클래스 이름에 맞게 수정

# 4. checkpoint 로드 (meta['CLASSES'] 오류 방지)
load_checkpoint(model, checkpoint_file, map_location='cpu')

# 5. CUDA 올리기 및 평가 모드
model.cuda()
model.eval()

# 6. 더미 입력 (512x512 RGB)
dummy_input = torch.randn(1, 3, 512, 512).cuda()

# 7. softmax confidence map 출력 래퍼 정의
class SegformerWithSoftmax(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model.encode_decode(x, img_metas=None)
        probs = F.softmax(logits, dim=1)
        return probs

# 8. 래핑 및 ONNX export
wrapped_model = SegformerWithSoftmax(model)

torch.onnx.export(
    wrapped_model,
    dummy_input,
    "segformer_confidence.onnx",  # 현재 폴더에 저장
    input_names=["input"],
    output_names=["confidence_map_tiles"],
    opset_version=13,
    dynamic_axes={
        "input": {0: "batch_size"},
        "confidence_map_tiles": {0: "batch_size"}
    }
)

print("✅ ONNX 변환 완료: segformer_confidence.onnx (softmax confidence map 포함)")
