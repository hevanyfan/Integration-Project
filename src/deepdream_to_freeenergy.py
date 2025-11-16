#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 DeepDream 输出 CSV 转换为 MOF-FreeEnergy/LLMProp 可用的候选集（包含 mofseq）。
完全 CPU 可跑；默认 tokenizer 为 't5-small'。

示例用法：
  python src/pipelines/deepdream_to_freeenergy.py \
      cp_max_1000_dream_results.csv \
      data/candidates_cp.csv \
      --tokenizer-name t5-small \
      --target-column dreamed_target \
      --top-k 200 \
      --drop-duplicates dreamed_mof_name,dreamed_edge_selfies \
      --max-length 512
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import pandas as pd
from transformers import AutoTokenizer

# SELFIES 可选依赖：不存在也能跑（只是无法把 SELFIES 解码为 SMILES）
try:
    import selfies as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None

# 关键：从项目内工具模块导入（需 PYTHONPATH=src）
from llmprop_utils import generate_mofseq

LOGGER = logging.getLogger("deepdream_to_freeenergy")

MOFNAME_L = "<mofname>"
MOFNAME_R = "</mofname>"
MOFID_L   = "<mofid>"
MOFID_R   = "</mofid>"


# ---------------------------
# 小工具函数
# ---------------------------

def _ensure_iterable(value: Optional[str]) -> List[str]:
    """把逗号分隔的字符串转为列表；空值返回空列表。"""
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def decode_selfies(selfies_string: str) -> Optional[str]:
    """尝试把 SELFIES 解码为 SMILES；若未安装 selfies，则返回 None。"""
    if not selfies_string:
        return None
    if sf is None:
        LOGGER.debug("SELFIES 未安装，跳过解码。")
        return None
    try:
        return sf.decoder(selfies_string)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("SELFIES 解码失败：%s | 错误：%s", selfies_string, exc)
        return None


def parse_topology_token(raw_token: str) -> str:
    """把形如 '[sxb]' 的拓扑 token 清洗为 'sxb'。"""
    if not raw_token:
        return "UNKNOWN"
    token = raw_token.strip().replace("[", "").replace("]", "").replace(" ", "")
    return token or "UNKNOWN"


def split_mofseq(mofseq: str) -> Tuple[Optional[str], Optional[str]]:
    """从已存在的 mofseq 中提取 <mofname> 与 <mofid> 的内容。"""
    import re
    re_mofname = re.compile(r"<mofname>(.*?)</mofname>", re.DOTALL)
    re_mofid   = re.compile(r"<mofid>(.*?)</mofid>",     re.DOTALL)
    if not mofseq:
        return None, None
    g = re_mofname.search(mofseq)
    l = re_mofid.search(mofseq)
    return (g.group(1) if g else None, l.group(1) if l else None)


def build_mofid_v1(row: pd.Series) -> str:
    """
    依据一行 DeepDream 结果构造一个“MOFid-v1 风格”的字符串。
    仅其“结构部分”最终会被下游清洗，但我们保留完整语义方便回溯。

    规则优先：
      1) 优先从 'dreamed_mof_string' 中解析结构与拓扑（按 '[&&]' 分割拓扑）。
      2) linker 取 'dreamed_edge_smiles'，缺失则解码 'dreamed_edge_selfies'。
      3) 若结构段里含 '[.]'，则尝试把两段分别解码（linker / sbu）。
      4) 兜底用 'dreamed_mof_name' 或 'UNKNOWN'。
    """
    mof_string = (row.get("dreamed_mof_string") or "") or ""
    edge_smiles = (row.get("dreamed_edge_smiles") or "") or ""
    edge_selfies = (row.get("dreamed_edge_selfies") or "") or ""

    structural_part = mof_string
    topology_token = ""
    if "[&&]" in mof_string:
        structural_part, topology_token = mof_string.rsplit("[&&]", maxsplit=1)
    topology = parse_topology_token(topology_token)

    linker_smiles = edge_smiles or decode_selfies(edge_selfies) or ""
    sbu_smiles = ""

    if "[.]" in structural_part:
        linker_part, sbu_part = structural_part.split("[.]", maxsplit=1)
        decoded_linker = decode_selfies(linker_part)
        if decoded_linker:
            linker_smiles = linker_smiles or decoded_linker
        sbu_smiles = decode_selfies(sbu_part) or sbu_part
    elif structural_part:
        sbu_smiles = decode_selfies(structural_part) or structural_part

    components = [c for c in (sbu_smiles, linker_smiles) if c]
    if not components:
        components.append(str(row.get("dreamed_mof_name", "UNKNOWN")))

    structural_descriptor = ".".join(components) or "UNKNOWN"

    topology_hint = topology
    if topology_hint == "UNKNOWN":
        dreamed_name = row.get("dreamed_mof_name")
        if isinstance(dreamed_name, str) and dreamed_name:
            topology_hint = dreamed_name.split("_")[0]
        elif isinstance(row.get("linker_hash"), str):
            topology_hint = row["linker_hash"]

    return f"mofid: {structural_descriptor} MOFid-v1.{topology_hint}"


def filter_dream_results(
    df_in: pd.DataFrame,
    *,
    min_target: Optional[float] = None,
    max_target: Optional[float] = None,
    target_column: str = "dreamed_target",
    top_k: Optional[int] = None,
    sort_by: Optional[str] = None,
    descending: bool = True,
    drop_duplicates: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """按阈值/排序/去重筛选候选。"""
    df = df_in.copy()

    if target_column in df.columns:
        if min_target is not None:
            df = df[df[target_column] >= min_target]
        if max_target is not None:
            df = df[df[target_column] <= max_target]
    else:
        LOGGER.info("列 '%s' 不存在，跳过数值过滤。", target_column)

    if drop_duplicates:
        df = df.drop_duplicates(subset=list(drop_duplicates))

    if top_k is not None and top_k > 0:
        sort_col = sort_by or target_column
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=not descending).head(top_k)
        else:
            LOGGER.warning("请求 top-k 但排序列 '%s' 不存在，将无排序取前 %d 条。", sort_col, top_k)
            df = df.head(top_k)

    return df.reset_index(drop=True)


def build_candidate_dataframe(
    dream_results: pd.DataFrame,
    *,
    fe_column: Optional[str] = None,
    fe_default: Optional[float] = None,
) -> pd.DataFrame:
    """把筛选后的 Dreaming 结果整理成下游需要的列：mof_name / mofid_v1 /（可选 FE_atom）。"""
    if "dreamed_mof_name" not in dream_results.columns:
        raise KeyError("缺少列 'dreamed_mof_name'，无法构建候选。")

    candidates = pd.DataFrame(
        {
            "mof_name": dream_results["dreamed_mof_name"],
            "mofid_v1": dream_results.apply(build_mofid_v1, axis=1),
        }
    )

    if fe_column and fe_column in dream_results.columns:
        candidates["FE_atom"] = dream_results[fe_column]
    else:
        candidates["FE_atom"] = fe_default

    return candidates


def enrich_with_mofseq(
    dataframe: pd.DataFrame,
    tokenizer_name: str,
    *,
    input_type: str = "mofseq",
    max_length: int = 512,
) -> pd.DataFrame:
    """
    用与训练一致的 tokenizer 生成/补齐 'mofseq' 表征。
    注意：generate_mofseq(df, tokenizer, input_type, max_length=...) 的第三个参数是位置参数。
    """
    LOGGER.info("加载分词器：%s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_tokens(["<mofname>", "</mofname>", "<mofid>", "</mofid>"])

    # 关键：第三个参数用“位置传递” input_type，避免错误的关键字参数名
    enriched = generate_mofseq(
        dataframe.copy(),
        tokenizer,
        input_type,
        max_length=max_length,
    )
    return enriched


def deepdream_to_freeenergy(
    input_path: Path,
    output_path: Path,
    *,
    tokenizer_name: str = "t5-small",
    min_target: Optional[float] = None,
    max_target: Optional[float] = None,
    top_k: Optional[int] = None,
    sort_by: Optional[str] = None,
    descending: bool = True,
    drop_duplicates: Optional[str] = None,
    target_column: str = "dreamed_target",
    fe_column: Optional[str] = None,
    fe_default: Optional[float] = None,
    max_length: int = 512,
) -> pd.DataFrame:
    """完整管线：读取→筛选→构造候选→生成 mofseq→写出 CSV。"""
    LOGGER.info("读取 DeepDream 结果：%s", input_path)
    dream_results = pd.read_csv(input_path)

    dup_cols = _ensure_iterable(drop_duplicates)
    filtered = filter_dream_results(
        dream_results,
        min_target=min_target,
        max_target=max_target,
        target_column=target_column,
        top_k=top_k,
        sort_by=sort_by,
        descending=descending,
        drop_duplicates=dup_cols,
    )
    LOGGER.info("筛选后保留 %d / %d 条候选。", len(filtered), len(dream_results))

    candidates = build_candidate_dataframe(
        filtered,
        fe_column=fe_column,
        fe_default=fe_default,
    )

    enriched = enrich_with_mofseq(
        candidates,
        tokenizer_name,
        input_type="mofseq",
        max_length=max_length,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    LOGGER.info("已保存候选集到：%s", output_path)

    return enriched


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepDream CSV → LLMProp 候选集（含 mofseq）")
    p.add_argument("input_csv",  type=Path, help="DeepDream 输出 CSV 路径")
    p.add_argument("output_csv", type=Path, help="写出的候选集 CSV 路径")
    p.add_argument("--tokenizer-name", default="t5-small",
                   help="HuggingFace 分词器名或本地路径（与训练一致）")
    p.add_argument("--target-column",  default="dreamed_target",
                   help="排序/阈值使用的目标列名")
    p.add_argument("--min-target", type=float, default=None, help="目标最小阈值")
    p.add_argument("--max-target", type=float, default=None, help="目标最大阈值")
    p.add_argument("--top-k", type=int, default=None, help="筛选排序后的前 K 条")
    p.add_argument("--sort-by", default=None, help="排序列；默认用 target 列")
    p.add_argument("--ascending", action="store_true", help="升序排序（默认降序）")
    p.add_argument("--drop-duplicates", default=None,
                   help="去重列名，逗号分隔，如 'dreamed_mof_name,dreamed_edge_selfies'")
    p.add_argument("--fe-column", default=None,
                   help="若已有自由能列名，写这里；否则用 --fe-default 兜底")
    p.add_argument("--fe-default", type=float, default=None,
                   help="自由能兜底值（若无 fe-column）")
    p.add_argument("--max-length", type=int, default=512,
                   help="生成 mofseq 时的最大 token 长度（与训练一致）")
    p.add_argument("--log-level", default="INFO", help="日志级别：INFO/DEBUG/...")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    deepdream_to_freeenergy(
        args.input_csv,
        args.output_csv,
        tokenizer_name=args.tokenizer_name,
        min_target=args.min_target,
        max_target=args.max_target,
        top_k=args.top_k,
        sort_by=args.sort_by,
        descending=not args.ascending,
        drop_duplicates=args.drop_duplicates,
        target_column=args.target_column,
        fe_column=args.fe_column,
        fe_default=args.fe_default,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
