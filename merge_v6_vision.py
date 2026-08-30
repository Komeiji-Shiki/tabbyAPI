from pathlib import Path
import json
import os
import shutil
import struct

from safetensors import safe_open
from safetensors.torch import save_file


TARGET = Path(r"models\qwen38-27b-exl3-3.5-v6")
SOURCE = Path(r"models\qwen38-v6-source")

VISION_PREFIX = "model.visual."
VISION_BITS = 6


def fail(msg):
    raise SystemExit(f"\n[ERROR] {msg}\n")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    temp = path.with_name(path.name + ".tmp")

    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    os.replace(temp, path)


def backup(path):
    bak = path.with_name(path.name + ".pre-v6.bak")

    if bak.exists():
        fail(
            f"备份已经存在：{bak}\n"
            "为了防止覆盖最初的文件，脚本停止。"
        )

    shutil.copy2(path, bak)
    return bak


def read_st_header(path):
    """
    只读取 safetensors header，
    不加载整个 tensor。
    """

    with path.open("rb") as f:
        raw = f.read(8)

        if len(raw) != 8:
            fail(f"无效的 safetensors：{path}")

        header_len = struct.unpack("<Q", raw)[0]
        header = json.loads(f.read(header_len))

    return header


def tensor_bytes(path, keys):
    header = read_st_header(path)

    total = 0

    for key in keys:
        if key not in header:
            fail(
                f"{path.name} 中找不到 index 声明的 tensor：\n"
                f"{key}"
            )

        start, end = header[key]["data_offsets"]
        total += end - start

    return total


def main():

    print()
    print("==============================================")
    print(" Qwen3.8-27B EXL3 3.50bpw + H6 + V6")
    print("==============================================")
    print()

    if not TARGET.is_dir():
        fail(f"目标目录不存在：{TARGET.resolve()}")

    if not SOURCE.is_dir():
        fail(f"V6 来源目录不存在：{SOURCE.resolve()}")

    target_index_path = TARGET / "model.safetensors.index.json"
    source_index_path = SOURCE / "model.safetensors.index.json"

    target_config_path = TARGET / "config.json"
    source_config_path = SOURCE / "config.json"

    for path in [
        target_index_path,
        source_index_path,
        target_config_path,
        source_config_path,
    ]:
        if not path.is_file():
            fail(f"缺少文件：{path}")

    target_index = load_json(target_index_path)
    source_index = load_json(source_index_path)

    target_map = target_index["weight_map"]
    source_map = source_index["weight_map"]

    # 找出旧 BF16 vision
    old_visual = {
        k: v
        for k, v in target_map.items()
        if k.startswith(VISION_PREFIX)
    }

    # 找出新的 V6 vision
    new_visual = {
        k: v
        for k, v in source_map.items()
        if k.startswith(VISION_PREFIX)
    }

    if not old_visual:
        fail("3.5 模型中没有找到 model.visual.*")

    if not new_visual:
        fail("V6 来源中没有找到 model.visual.*")

    target_shards = sorted(set(old_visual.values()))
    source_shards = sorted(set(new_visual.values()))

    # 当前官方两个分支都应该各自集中在一个 shard。
    # 如果未来官方重新分片，宁可停止也不冒险。
    if len(target_shards) != 1:
        fail(
            "3.5 模型的 vision 跨多个 shard：\n"
            + "\n".join(target_shards)
        )

    if len(source_shards) != 1:
        fail(
            "V6 vision 跨多个 shard：\n"
            + "\n".join(source_shards)
        )

    target_shard_name = target_shards[0]
    source_shard_name = source_shards[0]

    target_shard = TARGET / target_shard_name
    source_shard = SOURCE / source_shard_name

    if not target_shard.is_file():
        fail(f"缺少目标 shard：{target_shard}")

    if not source_shard.is_file():
        fail(
            f"缺少 V6 shard：{source_shard}\n\n"
            f"根据 index，需要下载：{source_shard_name}"
        )

    # 确认来源真的是 V6
    source_config = load_json(source_config_path)

    source_vision_bits = (
        source_config
        .get("quantization_config", {})
        .get("vision_bits")
    )

    if source_vision_bits != VISION_BITS:
        fail(
            f"来源 vision_bits = {source_vision_bits}\n"
            f"预期 = {VISION_BITS}"
        )

    print(f"3.5 vision shard : {target_shard_name}")
    print(f"V6 vision shard  : {source_shard_name}")
    print()

    print(f"原 vision tensor 数量 : {len(old_visual)}")
    print(f"V6 vision tensor 数量 : {len(new_visual)}")

    old_size = tensor_bytes(
        target_shard,
        old_visual.keys()
    )

    new_size = tensor_bytes(
        source_shard,
        new_visual.keys()
    )

    print(
        f"原 vision 数据大小 : "
        f"{old_size / 1024**3:.3f} GiB"
    )

    print(
        f"V6 vision 数据大小 : "
        f"{new_size / 1024**3:.3f} GiB"
    )

    print()
    print("[1/5] 读取 3.5 shard 的非视觉部分...")

    merged = {}

    with safe_open(
        str(target_shard),
        framework="pt",
        device="cpu"
    ) as f:

        original_keys = list(f.keys())
        metadata = f.metadata()

        for key in original_keys:

            if not key.startswith(VISION_PREFIX):
                merged[key] = f.get_tensor(key)

    print("[2/5] 加入 V6 vision tensor...")

    with safe_open(
        str(source_shard),
        framework="pt",
        device="cpu"
    ) as f:

        source_keys = set(f.keys())

        missing = set(new_visual) - source_keys

        if missing:
            fail(
                f"V6 shard 缺少 {len(missing)} 个 tensor。\n"
                "文件可能没有下载完整。"
            )

        for key in new_visual:
            merged[key] = f.get_tensor(key)

    expected_keys = (
        {
            k
            for k in original_keys
            if not k.startswith(VISION_PREFIX)
        }
        | set(new_visual)
    )

    if set(merged) != expected_keys:
        fail("待写入 tensor 集合校验失败。")

    temp_shard = target_shard.with_name(
        target_shard.name + ".v6tmp"
    )

    if temp_shard.exists():
        temp_shard.unlink()

    print("[3/5] 写入新 shard...")
    print("      这一步会写数 GB，请不要关闭窗口。")

    save_file(
        merged,
        str(temp_shard),
        metadata=metadata
    )

    del merged

    # 再验证一次新文件
    with safe_open(
        str(temp_shard),
        framework="pt",
        device="cpu"
    ) as f:
        written_keys = set(f.keys())

    if written_keys != expected_keys:
        temp_shard.unlink(missing_ok=True)
        fail(
            "新 shard 校验失败。\n"
            "原模型没有被修改。"
        )

    print("[4/5] 备份原始 BF16 vision shard...")

    shard_backup = target_shard.with_name(
        target_shard.name + ".pre-v6.bak"
    )

    if shard_backup.exists():

        temp_shard.unlink(missing_ok=True)

        fail(
            f"备份已经存在：{shard_backup}\n"
            "脚本不会覆盖它。"
        )

    # 原 shard -> backup
    os.replace(
        target_shard,
        shard_backup
    )

    # 新 shard -> 正式文件
    try:

        os.replace(
            temp_shard,
            target_shard
        )

    except Exception:

        # 出问题尽量自动恢复
        if (
            not target_shard.exists()
            and shard_backup.exists()
        ):
            os.replace(
                shard_backup,
                target_shard
            )

        raise

    # 备份 index 和 config
    index_backup = backup(target_index_path)
    config_backup = backup(target_config_path)

    print("[5/5] 更新 index 和 config.json...")

    # 删除旧 BF16 vision 的 index 项
    new_map = dict(target_map)

    for key in list(new_map):

        if key.startswith(VISION_PREFIX):
            del new_map[key]

    # 新 V6 tensor 全部放在重建后的目标 shard
    for key in new_visual:
        new_map[key] = target_shard_name

    target_index["weight_map"] = new_map

    # 修正 total_size
    old_total = int(
        target_index
        .get("metadata", {})
        .get("total_size", 0)
    )

    if old_total:

        target_index.setdefault(
            "metadata",
            {}
        )["total_size"] = (
            old_total
            - old_size
            + new_size
        )

    save_json(
        target_index_path,
        target_index
    )

    # config 保留原来的：
    # bits = 3.5
    # head_bits = 6
    # mtp_bits = 4
    # 只增加 vision_bits = 6

    target_config = load_json(
        target_config_path
    )

    qcfg = target_config.setdefault(
        "quantization_config",
        {}
    )

    qcfg["vision_bits"] = VISION_BITS

    save_json(
        target_config_path,
        target_config
    )

    print()
    print("==============================================")
    print(" 完成")
    print("==============================================")
    print()

    print(
        f"Text   : {qcfg.get('bits')} bpw"
    )

    print(
        f"Head   : {qcfg.get('head_bits')} bit"
    )

    print(
        f"MTP    : {qcfg.get('mtp_bits')} bit"
    )

    print(
        f"Vision : {qcfg.get('vision_bits')} bit"
    )

    print()
    print("原始文件备份：")
    print(shard_backup)
    print(index_backup)
    print(config_backup)

    print()
    print(
        "先用 TabbyAPI 加载并测试图片。"
        "确认正常后可以删除 *.pre-v6.bak。"
    )


if __name__ == "__main__":
    main()