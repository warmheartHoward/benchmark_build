# Museum Video Benchmark Builder

从博物馆展品视频中自动构建单帧图像世界知识识别 Benchmark。

## 流程

```
Excel输入 → 视频定位 → 抽帧 → Gemini选帧 → QA生成 → TSV输出
```

1. 读取 Excel：A列=博物馆文件夹名，B列=视频文件名，C列=GT标签（`skip` 则跳过）
2. 按配置的 FPS 对视频抽帧
3. 通过 OpenAI 兼容接口调用 Gemini，自动筛选清晰、无拖影、无铭牌的最佳帧
4. 根据 GT 标签反向生成识别类 QA
5. 输出 TSV 格式的 Benchmark 文件

## 安装

```bash
pip install -r requirements.txt
```

依赖：`openpyxl` `opencv-python` `openai` `pyyaml` `Pillow` `tqdm`

## 配置

编辑 `config.yaml`：

```yaml
# 输入
excel_path: "path/to/input.xlsx"
video_root: "path/to/video_root"

# 抽帧
frame_extraction:
  fps: 1          # 每秒抽几帧
  max_frames: 30  # 每个视频最多抽帧数

# API (OpenAI兼容接口)
api:
  base_url: "https://your-endpoint/v1"
  api_key: "your-key"
  model: "gemini-2.0-flash-lite"
```

完整配置项见 `config.yaml` 文件。

## 使用

```bash
# 标准运行
python main.py

# 命令行覆盖路径
python main.py --excel "D:/data/input.xlsx" --video-root "D:/data/videos"

# 仅抽帧，不调用API（测试用）
python main.py --dry-run
```

## 输入 Excel 格式

| A列（博物馆） | B列（视频名） | C列（GT） |
|---------------|--------------|----------|
| 故宫博物院     | bronze_ding.mp4 | 青铜鼎   |
| 国家博物馆     | jade_bi.mp4     | 玉璧     |
| 省博物馆       | silk_painting.mp4 | skip   |

## 输出 TSV 格式

| image_path | question | answer | museum | source_video |
|-----------|----------|--------|--------|-------------|
| output/selected/故宫博物院_青铜鼎_0.jpg | 图像中的文物是什么？ | 青铜鼎 | 故宫博物院 | bronze_ding.mp4 |

## 目录结构

```
Benchmark_build/
├── config.yaml          # 配置文件
├── main.py              # 主程序
├── requirements.txt     # 依赖
└── output/
    ├── frames/          # 抽帧结果
    ├── selected/        # 选中的最佳帧
    └── benchmark.tsv    # 最终输出
```
