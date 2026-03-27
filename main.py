"""
Benchmark Builder: 从博物馆视频构建单帧图像世界知识识别Benchmark
流程: Excel读取 -> 视频抽帧 -> Gemini选帧 -> QA生成 -> TSV输出
"""

import os
import sys
import csv
import yaml
import base64
import random
import logging
import argparse
from pathlib import Path

import cv2
import openpyxl
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_excel(excel_path: str) -> list[dict]:
    """读取Excel，返回 [{museum, video, gt}, ...]，跳过C列为skip的行"""
    wb = openpyxl.load_workbook(excel_path, read_only=True)
    ws = wb.active
    entries = []
    for row in ws.iter_rows(min_row=2, values_only=True):  # 跳过表头
        if len(row) < 3 or not row[0] or not row[1]:
            continue
        museum = str(row[0]).strip()
        video = str(row[1]).strip()
        gt = str(row[2]).strip() if row[2] else ""
        if gt.lower() == "skip":
            logger.info(f"跳过: {museum}/{video} (标记为skip)")
            continue
        entries.append({"museum": museum, "video": video, "gt": gt})
    wb.close()
    logger.info(f"从Excel读取到 {len(entries)} 条有效记录")
    return entries


def find_video_path(video_root: str, museum: str, video_name: str) -> str | None:
    """在 video_root/museum/ 下查找视频文件"""
    museum_dir = Path(video_root) / museum
    if not museum_dir.exists():
        # 尝试模糊匹配博物馆文件夹名
        for d in Path(video_root).iterdir():
            if d.is_dir() and museum in d.name:
                museum_dir = d
                break
        else:
            logger.warning(f"未找到博物馆文件夹: {museum}")
            return None

    # 递归搜索博物馆文件夹及所有子文件夹
    video_stem = Path(video_name).stem
    for dirpath, _, filenames in os.walk(museum_dir):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            # 精确匹配文件名
            if fname == video_name:
                return str(fpath)
            # 不带扩展名匹配
            if fpath.stem == video_stem:
                return str(fpath)
            # 部分匹配
            if video_name in fname:
                return str(fpath)

    logger.warning(f"未找到视频: {museum}/{video_name}")
    return None


def _cv_read_video(video_path: str) -> cv2.VideoCapture:
    """兼容中文路径的视频读取"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        # OpenCV 在 Windows 上不支持非ASCII路径，写入临时文件中转
        import tempfile
        with open(video_path, "rb") as stream:
            raw = stream.read()
        suffix = Path(video_path).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(raw)
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        cap._tmp_file = tmp.name  # 记录临时文件路径，后续清理
    return cap


def _cv_imwrite(path: str, img, params=None) -> bool:
    """兼容中文路径的图片写入"""
    if params is None:
        params = []
    # 先尝试直接写入
    success = cv2.imwrite(path, img, params)
    if not success:
        # 编码后用 Python IO 写入，绕过 OpenCV 的路径限制
        ext = Path(path).suffix
        ret, buf = cv2.imencode(ext, img, params)
        if ret:
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            success = True
    return success


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0,
                   max_frames: int = 30, img_format: str = "jpg", quality: int = 95) -> list[str]:
    """从视频中按指定fps抽帧，返回帧图片路径列表"""
    logger.info(f"正在打开视频: {video_path}")
    cap = _cv_read_video(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        logger.error("请确认已安装视频编解码器，或尝试: pip install opencv-python-headless")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"视频信息: fps={video_fps}, 总帧数={total_frames}")
    if video_fps <= 0 or total_frames <= 0:
        logger.error(f"视频信息异常: {video_path} (fps={video_fps}, frames={total_frames})")
        cap.release()
        return []

    # 计算抽帧间隔
    frame_interval = max(1, int(video_fps / fps))
    logger.info(f"抽帧间隔: 每 {frame_interval} 帧取1帧 (目标fps={fps})")
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0 and saved_count < max_frames:
            fname = f"frame_{frame_idx:06d}.{img_format}"
            fpath = os.path.join(output_dir, fname)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if img_format == "jpg" else []
            success = _cv_imwrite(fpath, frame, encode_params)
            if success:
                frame_paths.append(fpath)
                saved_count += 1
            else:
                logger.warning(f"写入帧失败: {fpath}")
        frame_idx += 1

    # 清理临时文件
    tmp_file = getattr(cap, "_tmp_file", None)
    cap.release()
    if tmp_file and os.path.exists(tmp_file):
        os.unlink(tmp_file)

    logger.info(f"从 {video_path} 抽取 {len(frame_paths)} 帧 -> {output_dir}")
    return frame_paths


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def select_best_frame(client: OpenAI, model: str, frame_paths: list[str],
                      gt_name: str, batch_size: int = 5, max_tokens: int = 1024) -> list[str]:
    """
    调用Gemini选出最适合作为benchmark的帧。
    返回被选中的帧路径列表。
    """
    if not frame_paths:
        return []

    selected_frames = []

    # 分批发送图片
    for batch_start in range(0, len(frame_paths), batch_size):
        batch = frame_paths[batch_start:batch_start + batch_size]
        content = [
            {
                "type": "text",
                "text": (
                    f"你是一个benchmark数据集构建助手。以下是从一个关于「{gt_name}」的博物馆展品视频中抽取的 {len(batch)} 帧图像。\n"
                    f"请从中选出最适合作为「单帧世界知识识别benchmark」的图像。\n\n"
                    f"选择标准：\n"
                    f"1. 图像清晰，不模糊\n"
                    f"2. 没有运动拖影\n"
                    f"3. 没有铭牌/标签/文字说明牌遮挡主体\n"
                    f"4. 展品主体完整可见，占据画面主要区域\n"
                    f"5. 光线良好，色彩自然\n"
                    f"6. 适合用于考察视觉识别能力（即仅凭图像内容就能辨认出是什么）\n\n"
                    f"可以选择多张符合条件的图像。\n"
                    f"请只返回被选中图像的编号（从1开始），用逗号分隔。例如：1,3,5\n"
                    f"如果没有合适的图像，返回：无"
                )
            }
        ]
        for i, fp in enumerate(batch):
            b64 = encode_image_base64(fp)
            ext = Path(fp).suffix.lstrip(".")
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            content.append({
                "type": "text",
                "text": f"图像 {i + 1}:"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"}
            })

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
            )
            answer = resp.choices[0].message.content.strip()
            logger.info(f"API返回选帧结果: {answer}")

            if "无" in answer:
                continue

            # 解析选中的编号
            for token in answer.replace("，", ",").split(","):
                token = token.strip().rstrip("。.）)")
                # 提取数字
                num_str = "".join(c for c in token if c.isdigit())
                if num_str:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(batch):
                        selected_frames.append(batch[idx])

        except Exception as e:
            logger.error(f"API调用失败: {e}")
            # 回退：选第一帧
            if batch:
                selected_frames.append(batch[0])

    if not selected_frames and frame_paths:
        logger.warning("未能选出合适帧，回退选取中间帧")
        selected_frames.append(frame_paths[len(frame_paths) // 2])

    return selected_frames


def generate_question(gt_name: str, language: str = "zh") -> str:
    """根据GT名称反向生成提问"""
    zh_templates = [
        "图像中展示的是什么？",
        "请识别图像中的物体。",
        "这是什么？",
        "图像中的展品是什么？",
        "请问图中展示的是哪种文物？",
        "你能辨认出图像中的物品吗？",
        "图像中的主体是什么？",
        "这张图片展示了什么？",
    ]

    en_templates = [
        "What is shown in the image?",
        "Identify the object in the image.",
        "What is this?",
        "What exhibit is displayed in the image?",
        "Can you recognize the item in the image?",
        "What is the main subject of this image?",
        "What does this image show?",
    ]

    # 根据GT内容选择更具针对性的模板
    zh_specific = {
        "动物": [
            "图像中的动物是什么？",
            "这是什么动物？",
            "请识别图中的动物种类。",
            "图像中展示的是哪种动物？",
        ],
        "植物": [
            "图像中的植物是什么？",
            "这是什么植物？",
            "请识别图中的植物种类。",
        ],
        "文物": [
            "图像中的文物是什么？",
            "这件文物是什么？",
            "请识别图中的文物。",
            "这是哪件文物？",
        ],
    }

    if language == "zh":
        # 尝试匹配特定类别模板
        for keyword, templates in zh_specific.items():
            if keyword in gt_name:
                return random.choice(templates)
        return random.choice(zh_templates)
    else:
        return random.choice(en_templates)


def save_selected_frame(src_path: str, selected_dir: str, museum: str,
                        gt_name: str, idx: int) -> str:
    """将选中帧复制到selected目录，返回新路径"""
    os.makedirs(selected_dir, exist_ok=True)
    ext = Path(src_path).suffix
    safe_name = gt_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    dst_name = f"{museum}_{safe_name}_{idx}{ext}"
    dst_path = os.path.join(selected_dir, dst_name)

    import shutil
    shutil.copy2(src_path, dst_path)
    return dst_path


def write_benchmark_tsv(results: list[dict], output_path: str):
    """输出benchmark TSV文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["image_path", "question", "answer", "museum", "source_video"])
        for r in results:
            writer.writerow([
                r["image_path"],
                r["question"],
                r["answer"],
                r["museum"],
                r["source_video"],
            ])
    logger.info(f"Benchmark已保存: {output_path} ({len(results)} 条)")


def main():
    parser = argparse.ArgumentParser(description="博物馆视频Benchmark构建工具")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--excel", default=None, help="Excel文件路径（覆盖配置）")
    parser.add_argument("--video-root", default=None, help="视频根目录（覆盖配置）")
    parser.add_argument("--output", default=None, help="输出TSV路径（覆盖配置）")
    parser.add_argument("--dry-run", action="store_true", help="只抽帧不调API")
    args = parser.parse_args()

    cfg = load_config(args.config)

    excel_path = args.excel or cfg["excel_path"]
    video_root = args.video_root or cfg["video_root"]
    output_tsv = args.output or cfg["benchmark_tsv"]
    frames_dir = cfg.get("frames_dir", "./output/frames")
    selected_dir = cfg.get("selected_dir", "./output/selected")

    fc = cfg.get("frame_extraction", {})
    fps = fc.get("fps", 1)
    max_frames = fc.get("max_frames", 30)
    img_format = fc.get("image_format", "jpg")
    quality = fc.get("quality", 95)

    api_cfg = cfg.get("api", {})
    qa_cfg = cfg.get("qa", {})
    language = qa_cfg.get("language", "zh")

    # 初始化API客户端
    client = None
    if not args.dry_run:
        client = OpenAI(
            base_url=api_cfg["base_url"],
            api_key=api_cfg["api_key"],
        )

    # 读取Excel
    entries = read_excel(excel_path)
    if not entries:
        logger.error("没有有效的数据条目")
        sys.exit(1)

    results = []

    for entry in tqdm(entries, desc="处理视频"):
        museum = entry["museum"]
        video_name = entry["video"]
        gt = entry["gt"]

        # 1. 查找视频
        video_path = find_video_path(video_root, museum, video_name)
        if not video_path:
            continue

        # 2. 抽帧
        video_frame_dir = os.path.join(frames_dir, museum, Path(video_name).stem)
        frame_paths = extract_frames(
            video_path, video_frame_dir,
            fps=fps, max_frames=max_frames,
            img_format=img_format, quality=quality
        )
        if not frame_paths:
            logger.warning(f"抽帧失败: {video_path}")
            continue

        # 3. 选帧
        if args.dry_run:
            selected = [frame_paths[len(frame_paths) // 2]]
            logger.info(f"[dry-run] 选择中间帧: {selected[0]}")
        else:
            selected = select_best_frame(
                client, api_cfg["model"], frame_paths, gt,
                batch_size=api_cfg.get("batch_size", 5),
                max_tokens=api_cfg.get("max_tokens", 1024),
            )

        # 4. 保存选中帧 + 生成QA
        for i, frame_path in enumerate(selected):
            saved_path = save_selected_frame(frame_path, selected_dir, museum, gt, i)
            question = generate_question(gt, language)
            results.append({
                "image_path": saved_path,
                "question": question,
                "answer": gt,
                "museum": museum,
                "source_video": video_name,
            })

    # 5. 输出TSV
    if results:
        write_benchmark_tsv(results, output_tsv)
    else:
        logger.warning("没有生成任何benchmark条目")

    logger.info("完成!")


if __name__ == "__main__":
    main()
