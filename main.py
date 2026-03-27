"""
Benchmark Builder: 从博物馆视频构建单帧图像世界知识识别Benchmark
流程: Excel读取 -> 视频抽帧 -> Gemini选帧 -> QA生成 -> TSV输出
"""

import os
import sys
import csv
import json
import yaml
import base64
import random
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import cv2
import numpy as np
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


def _calc_blur_score(frame) -> float:
    """计算图像模糊度分数（拉普拉斯方差），值越大越清晰"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0,
                   max_frames: int = 30, img_format: str = "jpg", quality: int = 95,
                   blur_threshold: float = 50.0) -> tuple[list[str], float]:
    """从视频中按指定fps抽帧，自动过滤模糊帧，返回 (帧图片路径列表, 视频原始fps)"""
    logger.info(f"正在打开视频: {video_path}")
    cap = _cv_read_video(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        logger.error("请确认已安装视频编解码器，或尝试: pip install opencv-python-headless")
        return [], 0.0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"视频信息: fps={video_fps}, 总帧数={total_frames}")
    if video_fps <= 0 or total_frames <= 0:
        logger.error(f"视频信息异常: {video_path} (fps={video_fps}, frames={total_frames})")
        cap.release()
        return [], 0.0

    # 计算抽帧间隔和目标帧号列表
    frame_interval = max(1, int(video_fps / fps))
    target_indices = [i * frame_interval for i in range(max_frames)
                      if i * frame_interval < total_frames]
    logger.info(f"计划抽取 {len(target_indices)} 帧 (间隔={frame_interval})")
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    skipped_blur = 0

    for target_idx in target_indices:
        # 直接 seek 到目标帧，避免逐帧读取
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # 本地模糊检测：过滤掉模糊帧，减少发给API的数量
        blur_score = _calc_blur_score(frame)
        if blur_score < blur_threshold:
            skipped_blur += 1
            continue

        fname = f"frame_{target_idx:06d}.{img_format}"
        fpath = os.path.join(output_dir, fname)
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if img_format == "jpg" else []
        success = _cv_imwrite(fpath, frame, encode_params)
        if success:
            frame_paths.append(fpath)
        else:
            logger.warning(f"写入帧失败: {fpath}")

    # 清理临时文件
    tmp_file = getattr(cap, "_tmp_file", None)
    cap.release()
    if tmp_file and os.path.exists(tmp_file):
        os.unlink(tmp_file)

    logger.info(f"从 {video_path} 抽取 {len(frame_paths)} 帧, "
                f"过滤模糊帧 {skipped_blur} 张 -> {output_dir}")
    return frame_paths, video_fps


def encode_image_base64(image_path: str, max_long_edge: int = 1024) -> str:
    """编码图片为base64，发送前缩小尺寸以加速API传输"""
    img = cv2.imread(image_path)
    if img is None:
        # 中文路径回退
        raw = Path(image_path).read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    # 无法解码时直接读取原始文件
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def select_frames(client: OpenAI, model: str, frame_paths: list[str],
                   gt_name: str, batch_size: int = 5,
                   max_tokens: int = 1024) -> tuple[str | None, list[str]]:
    """
    两阶段选帧：
      阶段1 - 每批筛选出所有合格帧（可多张）
      阶段2 - 从所有合格帧中用锦标赛选出唯一最佳帧
    返回: (best_frame_path, all_good_frame_paths)
    """
    if not frame_paths:
        return None, []

    def _ask_good_frames(candidates: list[str], round_name: str = "") -> list[str]:
        """从一组候选帧中让Gemini筛选出所有合格帧"""
        content = [
            {
                "type": "text",
                "text": (
                    f"你是一个benchmark数据集构建助手。以下是从一个关于「{gt_name}」的博物馆展品视频中抽取的 {len(candidates)} 帧图像。\n"
                    f"请筛选出所有适合作为「图像识别数据集」的图像。\n\n"
                    f"筛选标准：\n"
                    f"1. 图像清晰，不模糊\n"
                    f"2. 没有运动拖影\n"
                    f"3. 没有铭牌/标签/文字说明牌遮挡主体\n"
                    f"4. 展品主体完整可见\n"
                    f"5. 光线良好，色彩自然\n\n"
                    f"请返回所有合格图像的编号（从1开始），用逗号分隔。例如：1,3,5\n"
                    f"如果没有合格图像，返回：无"
                )
            }
        ]
        for i, fp in enumerate(candidates):
            b64 = encode_image_base64(fp)
            ext = Path(fp).suffix.lstrip(".")
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            content.append({"type": "text", "text": f"图像 {i + 1}:"})
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
            logger.info(f"API筛选结果{round_name}: {answer}")

            if "无" in answer:
                return []

            selected = []
            for token in answer.replace("，", ",").split(","):
                num_str = "".join(c for c in token if c.isdigit())
                if num_str:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(candidates):
                        selected.append(candidates[idx])
            return selected
        except Exception as e:
            logger.error(f"API调用失败{round_name}: {e}")
            return []

    def _ask_best_one(candidates: list[str], round_name: str = "") -> str | None:
        """从一组候选帧中让Gemini选出最佳的1帧"""
        content = [
            {
                "type": "text",
                "text": (
                    f"你是一个benchmark数据集构建助手。以下是从一个关于「{gt_name}」的博物馆展品视频中筛选出的 {len(candidates)} 帧合格图像。\n"
                    f"请从中选出**最适合**作为「单帧世界知识识别benchmark测试题」的**唯一一张**图像。\n\n"
                    f"选择标准（按优先级排序）：\n"
                    f"1. 图像最清晰锐利\n"
                    f"2. 展品主体占比最大、最完整\n"
                    f"3. 没有任何文字/铭牌干扰\n"
                    f"4. 最适合考察视觉识别能力（仅凭图像就能辨认出是什么）\n\n"
                    f"请只返回最佳图像的编号（从1开始），只返回一个数字，不要解释。"
                )
            }
        ]
        for i, fp in enumerate(candidates):
            b64 = encode_image_base64(fp)
            ext = Path(fp).suffix.lstrip(".")
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            content.append({"type": "text", "text": f"图像 {i + 1}:"})
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
            logger.info(f"API选帧结果{round_name}: {answer}")

            num_str = "".join(c for c in answer if c.isdigit())
            if num_str:
                idx = int(num_str) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception as e:
            logger.error(f"API调用失败{round_name}: {e}")

        return None

    # ===== 阶段1：分批筛选所有合格帧 =====
    all_good_frames = []
    for batch_start in range(0, len(frame_paths), batch_size):
        batch = frame_paths[batch_start:batch_start + batch_size]
        good = _ask_good_frames(batch, f" [批次{batch_start // batch_size + 1}]")
        all_good_frames.extend(good)

    if not all_good_frames:
        logger.warning("所有批次均未筛选出合格帧，回退选取中间帧")
        fallback = frame_paths[len(frame_paths) // 2]
        return fallback, [fallback]

    logger.info(f"阶段1完成: 从 {len(frame_paths)} 帧中筛选出 {len(all_good_frames)} 帧合格")

    # ===== 阶段2：锦标赛选出最佳1帧 =====
    if len(all_good_frames) == 1:
        return all_good_frames[0], all_good_frames

    # 如果合格帧数量不多，直接一次PK
    if len(all_good_frames) <= batch_size:
        best = _ask_best_one(all_good_frames, " [决赛]")
        if best:
            return best, all_good_frames
        return all_good_frames[0], all_good_frames

    # 合格帧多时，分批PK再决赛
    batch_winners = []
    for batch_start in range(0, len(all_good_frames), batch_size):
        batch = all_good_frames[batch_start:batch_start + batch_size]
        winner = _ask_best_one(batch, f" [半决赛{batch_start // batch_size + 1}]")
        if winner:
            batch_winners.append(winner)

    if not batch_winners:
        return all_good_frames[0], all_good_frames

    if len(batch_winners) == 1:
        return batch_winners[0], all_good_frames

    final = _ask_best_one(batch_winners, " [决赛]")
    return (final or batch_winners[0]), all_good_frames


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


def _extract_frame_timestamp(frame_path: str, video_fps: float) -> str:
    """从帧文件名中提取帧号并转为时间戳字符串，如 '00m12s'"""
    stem = Path(frame_path).stem  # e.g. "frame_000060"
    num_str = "".join(c for c in stem if c.isdigit())
    if num_str and video_fps > 0:
        frame_idx = int(num_str)
        total_sec = frame_idx / video_fps
        minutes = int(total_sec // 60)
        seconds = int(total_sec % 60)
        return f"{minutes:02d}m{seconds:02d}s"
    return "00m00s"


def _make_filename(museum: str, video_name: str, timestamp: str, gt_name: str) -> str:
    """生成标准文件名: 博物馆_视频名_帧时刻_文物名"""
    safe_museum = museum.replace("/", "_").replace("\\", "_").replace(" ", "_")
    video_stem = Path(video_name).stem
    safe_gt = gt_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{safe_museum}_{video_stem}_{timestamp}_{safe_gt}"


def save_selected_frame(src_path: str, dst_dir: str, museum: str, video_name: str,
                        gt_name: str, video_fps: float, idx: int = 0) -> str:
    """将选中帧复制到目标目录，文件名格式: 博物馆_视频名_帧时刻_文物名"""
    import shutil
    os.makedirs(dst_dir, exist_ok=True)
    ext = Path(src_path).suffix
    timestamp = _extract_frame_timestamp(src_path, video_fps)
    base_name = _make_filename(museum, video_name, timestamp, gt_name)
    # 多帧时用后缀区分
    suffix = f"_{idx}" if idx > 0 else ""
    dst_path = os.path.join(dst_dir, f"{base_name}{suffix}{ext}")
    shutil.copy2(src_path, dst_path)
    return dst_path


def save_frame_with_json(src_path: str, dst_dir: str, video_name: str,
                         gt_name: str, video_fps: float, question: str,
                         museum: str) -> str:
    """保存最佳帧并生成同名JSON文件，返回图片路径"""
    import shutil
    os.makedirs(dst_dir, exist_ok=True)
    ext = Path(src_path).suffix
    timestamp = _extract_frame_timestamp(src_path, video_fps)
    base_name = _make_filename(museum, video_name, timestamp, gt_name)

    img_path = os.path.join(dst_dir, f"{base_name}{ext}")
    json_path = os.path.join(dst_dir, f"{base_name}.json")

    shutil.copy2(src_path, img_path)

    qa_data = {
        "image": f"{base_name}{ext}",
        "question": question,
        "answer": gt_name,
        "museum": museum,
        "source_video": video_name,
        "timestamp": timestamp,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    return img_path


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


def process_single_entry(entry: dict, *, video_root: str, frames_dir: str,
                         selected_dir: str, fps: float, max_frames: int,
                         img_format: str, quality: int, blur_threshold: float,
                         client: OpenAI | None, api_cfg: dict, language: str,
                         dry_run: bool, done_set: set | None = None,
                         done_lock: threading.Lock | None = None) -> list[dict]:
    """处理单个视频条目，返回该视频生成的benchmark结果列表（线程安全）"""
    museum = entry["museum"]
    video_name = entry["video"]
    gt = entry["gt"]

    # 断点续传：跳过已处理的条目
    entry_key = f"{museum}/{video_name}"
    if done_set is not None and entry_key in done_set:
        logger.info(f"跳过已处理: {entry_key}")
        return []

    # 1. 查找视频
    video_path = find_video_path(video_root, museum, video_name)
    if not video_path:
        return []

    # 2. 抽帧
    video_frame_dir = os.path.join(frames_dir, museum, Path(video_name).stem)
    frame_paths, video_fps = extract_frames(
        video_path, video_frame_dir,
        fps=fps, max_frames=max_frames,
        img_format=img_format, quality=quality,
        blur_threshold=blur_threshold
    )
    if not frame_paths:
        logger.warning(f"抽帧失败: {video_path}")
        return []

    # 3. 选帧
    if dry_run:
        best_frame = frame_paths[len(frame_paths) // 2]
        all_good_frames = frame_paths
        logger.info(f"[dry-run] 选择中间帧: {best_frame}")
    else:
        best_frame, all_good_frames = select_frames(
            client, api_cfg["model"], frame_paths, gt,
            batch_size=api_cfg.get("batch_size", 5),
            max_tokens=api_cfg.get("max_tokens", 1024),
        )

    if not best_frame:
        logger.warning(f"未能选出合适帧: {museum}/{video_name}")
        return []

    # 4a. 保存最佳帧 + 同名JSON 到 benchmark 目录
    question = generate_question(gt, language)
    saved_best = save_frame_with_json(
        best_frame, selected_dir, video_name, gt, video_fps,
        question=question, museum=museum,
    )
    entry_results = [{
        "image_path": saved_best,
        "question": question,
        "answer": gt,
        "museum": museum,
        "source_video": video_name,
    }]

    # 4b. 保存所有合格帧到 candidates 目录（可用于训练等）
    candidates_dir = os.path.join(os.path.dirname(selected_dir), "candidates", museum)
    for i, fp in enumerate(all_good_frames):
        save_selected_frame(fp, candidates_dir, museum, video_name, gt, video_fps, idx=i)
    logger.info(f"保存 {len(all_good_frames)} 张合格帧 -> {candidates_dir}")

    # 标记已完成，用于断点续传
    if done_set is not None and done_lock is not None:
        with done_lock:
            done_set.add(entry_key)

    return entry_results


def main():
    parser = argparse.ArgumentParser(description="博物馆视频Benchmark构建工具")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--excel", default=None, help="Excel文件路径（覆盖配置）")
    parser.add_argument("--video-root", default=None, help="视频根目录（覆盖配置）")
    parser.add_argument("--output", default=None, help="输出TSV路径（覆盖配置）")
    parser.add_argument("--dry-run", action="store_true", help="只抽帧不调API")
    parser.add_argument("--workers", type=int, default=None, help="并行线程数（覆盖配置）")
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
    blur_threshold = fc.get("blur_threshold", 50.0)

    api_cfg = cfg.get("api", {})
    qa_cfg = cfg.get("qa", {})
    language = qa_cfg.get("language", "zh")
    max_workers = args.workers or cfg.get("max_workers", 4)

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

    # 断点续传：从已有TSV中读取已完成的条目
    done_set: set[str] = set()
    done_lock = threading.Lock()
    results = []
    if os.path.exists(output_tsv):
        with open(output_tsv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                done_set.add(f"{row['museum']}/{row['source_video']}")
                results.append(row)
        logger.info(f"断点续传: 已有 {len(done_set)} 条已处理记录，跳过这些视频")

    results_lock = threading.Lock()

    common_kwargs = dict(
        video_root=video_root, frames_dir=frames_dir, selected_dir=selected_dir,
        fps=fps, max_frames=max_frames, img_format=img_format, quality=quality,
        blur_threshold=blur_threshold, client=client, api_cfg=api_cfg,
        language=language, dry_run=args.dry_run,
        done_set=done_set, done_lock=done_lock,
    )

    logger.info(f"启动 {max_workers} 个工作线程处理 {len(entries)} 个视频")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {
            executor.submit(process_single_entry, entry, **common_kwargs): entry
            for entry in entries
        }
        with tqdm(total=len(entries), desc="处理视频") as pbar:
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    entry_results = future.result()
                    if entry_results:
                        with results_lock:
                            results.extend(entry_results)
                except Exception as e:
                    logger.error(f"处理失败 {entry['museum']}/{entry['video']}: {e}")
                pbar.update(1)

    # 5. 输出TSV
    if results:
        write_benchmark_tsv(results, output_tsv)
    else:
        logger.warning("没有生成任何benchmark条目")

    logger.info(f"完成! 共生成 {len(results)} 条benchmark条目")


if __name__ == "__main__":
    main()
