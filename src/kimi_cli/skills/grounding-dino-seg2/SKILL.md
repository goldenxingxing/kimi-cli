---
name: grounding-dino-seg2
description: "基于 Grounding DINO + SAM 的零样本精确抠图，自动检测目标物体、生成像素级分割掩码、将背景替换为指定颜色（或透明），并裁剪成恰好覆盖完整主体的最小画布。当用户需要'抠图'、'去背景'、'换背景颜色'、'把图中的XX主体提取成透明底/白底/纯色背景'、'精确分割并裁剪主体'、'制作透明PNG'、'把商品从图中分割出来'时触发。"
---

# grounding-dino-seg

基于 [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) + [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) 的零样本精确抠图工具。

核心流程：**文本检测（DINO）→ 像素级分割（SAM）→ 背景替换 → 最小主体裁剪**

- **Grounding DINO**：通过自然语言找到目标的边界框
- **SAM**：以边界框为 prompt，生成精确到像素的分割掩码
- **背景替换**：将掩码外的区域替换为指定颜色或透明通道
- **主体裁剪**：自动计算掩码的最紧边界框，输出恰好覆盖完整主体的图片

---

## 快速开始

### 命令行方式

```bash
# 抠出猫咪，白色背景，裁剪到最小主体区域
python scripts/seg_bg_replace.py \
  --image input.jpg \
  --text "a cat" \
  --output cat_white_bg.jpg \
  --bg-color "255,255,255"

# 透明底 PNG（适合后续合成）
python scripts/seg_bg_replace.py \
  --image photo.jpg \
  --text "the main product" \
  --output product_transparent.png \
  --bg-color transparent

# 黑色背景，保留完整原图尺寸（不裁剪）
python scripts/seg_bg_replace.py \
  --image input.jpg \
  --text "a person" \
  --output person_black_bg.jpg \
  --bg-color "0,0,0" \
  --no-crop

# 裁剪时留出 20px 边距
python scripts/seg_bg_replace.py \
  --image input.jpg \
  --text "a handbag" \
  --output bag_crop.png \
  --bg-color transparent \
  --padding 20
```

### Python API 方式

```python
from seg_bg_replace import seg_and_replace

result = seg_and_replace(
    image_path="input.jpg",
    text="a cat",
    output_path="output.png",
    bg_color=None,            # None = 透明底；(255,255,255) = 白底
    crop=True,                # 是否裁剪到最小主体
    padding=10,               # 主体边距（像素）
    multi=False,              # False = 只处理置信度最高的检测框
    threshold=0.3,
    text_threshold=0.25,
    dino_model_id="IDEA-Research/grounding-dino-tiny",
    sam_model_id="facebook/sam-vit-base",
    device="auto",
)
print(result)
```

---

## 参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--image` / `image_path` | `str` | ✅ | - | 输入图片路径，支持 jpg、png、webp 等常见格式 |
| `--text` | `str` | ✅ | - | 目标物体描述，如 `"a cat"`、`"the main product"` |
| `--output` / `output_path` | `str` | ✅ | - | 输出图片路径。透明度需存为 `.png` |
| `--bg-color` / `bg_color` | `str`/`tuple` | ❌ | `"255,255,255"` | 背景颜色。`"transparent"` 输出 RGBA PNG；`"R,G,B"` 输出纯色背景 |
| `--no-crop` / `crop=False` | `bool` | ❌ | `False` | 跳过裁剪步骤，输出与原图等大的图片 |
| `--padding` / `padding` | `int` | ❌ | `0` | 裁剪时在主体四周保留的额外像素边距 |
| `--multi` / `multi` | `bool` | ❌ | `False` | 将所有检测框的掩码合并（适用于多目标场景） |
| `--threshold` | `float` | ❌ | `0.3` | DINO 检测置信度阈值 |
| `--text-threshold` | `float` | ❌ | `0.25` | DINO 文本匹配阈值 |
| `--dino-model` / `dino_model_id` | `str` | ❌ | `"IDEA-Research/grounding-dino-tiny"` | Grounding DINO 模型 ID |
| `--sam-model` / `sam_model_id` | `str` | ❌ | `"facebook/sam-vit-base"` | SAM 模型 ID，可选 `sam-vit-huge`（更精准但更慢） |
| `--device` | `str` | ❌ | `"auto"` | 运行设备，自动选择 CUDA/MPS/CPU |
| `--json-output` | `str` | ❌ | `None` | 将元数据写入 JSON 文件 |

---

## 输出格式

### 图片输出

- **背景色模式**（`--bg-color "R,G,B"`）：输出 RGB 图片，背景为纯色
- **透明底模式**（`--bg-color transparent`）：输出 RGBA PNG，背景为透明通道
- 裁剪后图片尺寸 = 主体最小边界框 + `padding`

### JSON 元数据（`--json-output`）

```json
{
  "success": true,
  "output_path": "output.png",
  "output_size": [320, 480],
  "detections": [
    {
      "box": [120.0, 80.0, 440.0, 560.0],
      "score": 0.87,
      "label": "a cat"
    }
  ],
  "mask_coverage": 0.312
}
```

`mask_coverage` 表示主体像素占原图总像素的比例，可用于质量判断。

---

## 依赖安装

```bash
# PyTorch（根据 CUDA 版本调整）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Transformers（含 Grounding DINO 和 SAM 支持）
pip install transformers>=4.38 accelerate

# 图像处理
pip install Pillow numpy
```

### 模型大小参考

| 模型 | 大小 | 说明 |
|------|------|------|
| `grounding-dino-tiny` | ~172 MB | 速度快，适合大多数场景 |
| `grounding-dino-base` | ~341 MB | 精度更高 |
| `sam-vit-base` | ~375 MB | 平衡速度与精度（推荐） |
| `sam-vit-large` | ~1.2 GB | 更精细的边缘分割 |
| `sam-vit-huge` | ~2.4 GB | 最高精度 |

首次运行会自动从 HuggingFace 下载并缓存模型。

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 分割边缘不干净（锯齿/漏掉细节） | 换用 `sam-vit-large` 或 `sam-vit-huge` |
| 检测不到目标 | 降低 `--threshold` 和 `--text-threshold`；使用更具体的文本描述 |
| 背景被误判为主体 | 提高 `--threshold`；使用更精确的文本描述 |
| 多个物体时只想保留一个 | 默认 `multi=False` 只处理置信度最高的框，保持此默认值 |
| 透明底图在浏览器显示为白色 | 正常现象，白色是透明通道的默认填充色，实际通道数据正确 |
| 主体被裁过紧 | 增加 `--padding` 参数（如 `--padding 20`） |
| 显存不足 | 使用 `--device cpu`，或换用更小的 DINO/SAM 模型 |

---

## 触发关键词

当用户表达以下意图时，应触发此 Skill：
- "帮我抠图，把背景换成白色"
- "把图中的商品提取成透明底 PNG"
- "精确分割猫咪并去掉背景"
- "把这张商品图的背景换成纯黑色"
- "裁剪出图中人物主体，背景改成蓝色"
- "制作纯白底的商品图"
- "把图中的 XX 抠出来，背景透明"
- "分割主体并裁剪成最小画布"
