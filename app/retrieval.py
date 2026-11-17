"""第5回: ページ単位のベクトル検索。"""

import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from collections import defaultdict, deque
from pathlib import Path

from google import genai
from google.genai import types
from pydantic_ai.exceptions import ModelHTTPError
import sqlite_vec

from app.config import (
    EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL,
    RAG_DB, RAG_DOCUMENT_IDS, RAG_SOURCE, RAG_TOP_K, RAG_TREE, get_api_key,
    RAG_GRAPH, GRAPH_START_K, GRAPH_HOPS, GRAPH_MAX_RESULTS,
    HYBRID_SEED_K, HYBRID_MAX_SECTION_SEEDS, HYBRID_MAX_ADJACENT,
    HYBRID_MAX_RESULTS, HYBRID_TERM_MAX_DEGREE,
)

_HEADER_LINE_RE = re.compile(
    r"^(第\s*\d+\s*回|\d+/\d+|part\s*\d+|appendix|まとめ|演習|クイズ\s*\d+(/\d+)?|"
    r"補足|付録|次回予告|今日の構成|今日の問い|前回の振り返り|自己紹介|アジェンダ)\b",
    re.IGNORECASE,
)


def get_all_materials() -> str:
    """全資料を、講義・章・ページの情報付きで1つの文字列にまとめる。"""
    documents = json.loads(RAG_SOURCE.read_text(encoding="utf-8"))
    return "\n\n".join(
        f"資料: {document['id']} / {document['title']}\n"
        f"章: {section['section']}\n"
        f"ページ: PDF p.{page['page']} / {page['title']}\n{page['content']}"
        for document in documents
        for section in document["content"]
        for page in section["content"]
    )

def load_pages(source: Path = RAG_SOURCE) -> list[dict]:
    """資料→章→ページのJSONを、対象資料のページ一覧に変換する。"""
    documents = json.loads(source.read_text(encoding="utf-8"))
    pages = []
    for document in documents:
        if document["id"] not in RAG_DOCUMENT_IDS:
            continue
        for section in document["content"]:
            for page in section["content"]:
                pages.append({"page_id": page["id"], "document_id": document["id"],
                              "lecture": int(document["id"][3:5]), "document_title": document["title"],
                              "section": section["section"], "title": page["title"],
                              "page": page["page"], "content": page["content"]})
    return pages


def browse_tree(node_id: str | None = None) -> dict:
    """直下のページは先頭文、指定したページ自身は全文を返す。"""
    tree = json.loads(RAG_TREE.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            info.update({key: value for key, value in pages[node["id"]].items() if key != "page_id"})
            if not full:
                info["content"] = re.split(r"(?<=[。！？])|\n", info["content"].strip(), maxsplit=1)[0]
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


def read_page(page_id: str) -> dict:
    """第1〜3回の本講義から、指定した1ページの本文と引用情報を返す。"""
    for page in load_pages(RAG_SOURCE):
        if page["page_id"] == page_id:
            return page
    raise ValueError("対象のページが見つかりません。")


def normalized_vector(values: list[float]) -> list[float]:
    """ベクトルの長さを1に揃える。"""
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("埋め込みがゼロベクトルです。")
    return [value / norm for value in values]


def embed_text(text: str, task_type: str) -> list[float]:
    """Geminiで文章を埋め込む。入力上限を超えた場合は切り捨てず停止する。"""
    with genai.Client(api_key=get_api_key()) as client:
        count = client.models.count_tokens(model=EMBEDDING_MODEL, contents=text).total_tokens
        if count is None or count > EMBEDDING_MAX_TOKENS:
            raise ValueError(f"埋め込み入力が上限を超えるか計測不能です: {count} tokens。")
        response = client.models.embed_content(
            model=EMBEDDING_MODEL, contents=text,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
        )
    embedding = response.embeddings[0]
    if embedding.statistics and embedding.statistics.truncated:
        raise ValueError("埋め込みAPIが本文を切り捨てました。")
    return normalized_vector(embedding.values)


@contextmanager
def open_index(path: Path):
    """既存の検索DBを読み取り専用で開き、終了時に閉じる。"""
    if not path.is_file():
        raise FileNotFoundError("検索DBがありません。")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        yield db
    finally:
        db.close()


def search(query: str) -> list[dict]:
    """質問を埋め込み、近いページを最大RAG_TOP_K件返す。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")
    with open_index(RAG_DB) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute("""
            SELECT page_id, document_id, lecture, document_title, section, title, page, content,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id LIMIT ?
        """, (sqlite_vec.serialize_float32(vector), RAG_TOP_K))
        return [dict(row) for row in rows]


def search_graph(query: str) -> dict:
    """ベクトル検索を入口に、前後ページと共通用語で候補を広げる。"""
    pages = {page["page_id"]: page for page in load_pages(RAG_SOURCE)}
    graph = json.loads(RAG_GRAPH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]
             if node["kind"] == "term" or node["id"] in pages}
    adjacent = defaultdict(set)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if edge["relation"] in ("next_page", "mentions") and a in nodes and b in nodes:
            adjacent[a].add(b)
            adjacent[b].add(a)

    seeds = [hit for hit in search(query) if hit["page_id"] in pages][:min(GRAPH_START_K, GRAPH_MAX_RESULTS)]
    seed_ids = {seed["page_id"] for seed in seeds}
    results, candidates = [], {}
    for rank, seed in enumerate(seeds):
        root = seed["page_id"]
        results.append({**pages[root], "distance": seed["distance"], "hops": 0,
                        "path": [{"id": root, "title": pages[root]["title"]}]})
        queue = deque([(root, [root])])
        visited = {root}
        while queue:
            current, path = queue.popleft()
            if len(path) - 1 >= GRAPH_HOPS:
                continue
            # 同じ深さなら、接続先が少ない用語からたどる。
            neighbors = sorted(adjacent[current], key=lambda n: (len(adjacent[n]), n))
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                route = path + [neighbor]
                queue.append((neighbor, route))
                if neighbor not in pages or neighbor in seed_ids:
                    continue
                frequency = max((len(adjacent[n]) for n in route if nodes[n]["kind"] == "term"), default=0)
                priority = (len(route) - 1, frequency, rank, neighbor)
                candidate = {**pages[neighbor], "hops": len(route) - 1,
                             "path": [{"id": n, "title": nodes[n]["title"]} for n in route]}
                if neighbor not in candidates or priority < candidates[neighbor][0]:
                    candidates[neighbor] = (priority, candidate)
    expanded = [candidate for _, candidate in sorted(candidates.values())]
    return {"seeds": results, "expanded": expanded[:max(0, GRAPH_MAX_RESULTS - len(results))],
            "max_hops": GRAPH_HOPS}


def _compact_norm(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _body_lines(page: dict, include_citations: bool = False) -> list[str]:
    norm_title = _compact_norm(page.get("title", ""))
    norm_sec = _compact_norm(page.get("section", ""))
    lines = []
    for raw in page.get("content", "").splitlines():
        line = raw.strip()
        if not line or line.startswith(("©", "(c)", "【図表")):
            continue
        if not include_citations and line.startswith(("出典：", "出典）", "情報確認日：")):
            continue
        norm_line = _compact_norm(line)
        if norm_line in (norm_title, norm_sec):
            continue
        if len(line) <= 36 and (_HEADER_LINE_RE.match(line) or (norm_sec and norm_sec in norm_line)):
            continue
        lines.append(line)
    return lines


def is_substantive_page(page: dict) -> bool:
    """章扉など本文がほぼ見出しのみのスライドを除外判定する。"""
    body_chars = sum(len(line) for line in _body_lines(page, include_citations=False))
    return body_chars >= 20


def extract_lead_summary(page: dict, max_chars: int = 90) -> str:
    """タイトルや章見出しの重複行を除き、スライドの実質的な先頭要約文を返す。"""
    candidates = _body_lines(page, include_citations=False) or _body_lines(page, include_citations=True)
    if not candidates:
        first = re.split(r"(?<=[。！？])|\n", page.get("content", "").strip(), maxsplit=1)[0].strip()
        return first[:max_chars]
    lead = re.split(r"(?<=[。！？])", candidates[0], maxsplit=1)[0].strip()
    if len(lead) < 16 and len(candidates) >= 2:
        second = re.split(r"(?<=[。！？])", candidates[1], maxsplit=1)[0].strip()
        lead = f"{lead} / {second}"
    return lead[:max_chars] + ("…" if len(lead) > max_chars else "")


def browse_outline(node_id: str | None = None, source: Path = RAG_SOURCE, tree_path: Path = RAG_TREE) -> dict:
    """見出し重複を除いた実質先頭文付きの目次を返し、ページ指定時は全文を返す。"""
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in RAG_DOCUMENT_IDS]
    node_id = node_id or tree["id"]
    pages = {page["page_id"]: page for page in load_pages(source)}

    def describe(node, full=False):
        info = {key: node[key] for key in ("id", "kind", "title")}
        if node["kind"] == "page":
            page_data = pages[node["id"]]
            info.update({key: value for key, value in page_data.items() if key != "page_id"})
            if not full:
                info["content"] = extract_lead_summary(page_data)
        return info

    remaining = [tree]
    while remaining:
        node = remaining.pop()
        if node["id"] == node_id:
            return {
                "node": describe(node, full=True),
                "children": [describe(child) for child in node.get("children", [])],
            }
        remaining.extend(node.get("children", []))
    raise ValueError("対象の目次ノードが見つかりません。")


_TOKEN_RE = re.compile(r"[A-Za-z0-9_./+-]{2,}|[ァ-ヶー]{2,}|[一-龥]{2,}")


def _lexical_idf_scores(query: str, pages: dict[str, dict]) -> dict[str, float]:
    """固有名詞や数値表現の取りこぼしを防ぐため、214ページ上のIDF重み付き語彙一致スコア（0〜1正規化）を計算する。"""
    raw_tokens = {tok.casefold() for tok in _TOKEN_RE.findall(query)}
    if not raw_tokens:
        return {pid: 0.0 for pid in pages}

    n_docs = max(1, len(pages))
    doc_texts = {
        pid: f"{p['title']}\n{p['section']}\n{p['content']}".casefold()
        for pid, p in pages.items()
    }
    token_weights = {}
    for tok in raw_tokens:
        df = sum(1 for text in doc_texts.values() if tok in text)
        if 1 <= df < n_docs:
            token_weights[tok] = math.log((n_docs + 1.0) / (df + 0.5))

    total_weight = sum(token_weights.values())
    if total_weight <= 0:
        return {pid: 0.0 for pid in pages}
    return {
        pid: sum(w for tok, w in token_weights.items() if tok in text) / total_weight
        for pid, text in doc_texts.items()
    }


def search_hybrid(
    query: str,
    source: Path = RAG_SOURCE,
    graph_path: Path = RAG_GRAPH,
    db_path: Path = RAG_DB,
) -> dict:
    """密ベクトル＋IDF語彙一致の多様なシードに、章内前後ページと特異度重み付き用語・関連用語グラフ展開を統合する。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("検索語が空です。")

    pages = {page["page_id"]: page for page in load_pages(source)}
    substantive_ids = {pid for pid, page in pages.items() if is_substantive_page(page)} or set(pages)
    lex_scores = _lexical_idf_scores(query, pages)

    with open_index(db_path) as db:
        vector = embed_text(query.strip(), "RETRIEVAL_QUERY")
        rows = db.execute(
            """
            SELECT page_id, vec_distance_cosine(embedding, ?) AS distance
            FROM chunks ORDER BY distance, page_id
            """,
            (sqlite_vec.serialize_float32(vector),),
        ).fetchall()

    distances = {row["page_id"]: float(row["distance"]) for row in rows if row["page_id"] in pages}

    def hybrid_rel(pid: str) -> float:
        return (1.0 - distances.get(pid, 1.0)) + 0.25 * lex_scores.get(pid, 0.0)

    ranked_hits = sorted(distances.keys(), key=lambda pid: (-hybrid_rel(pid), pid))

    # 1. 同一章への偏りを抑えつつ、実質本文を持つ上位シードを選ぶ
    seeds_ids = []
    section_counts = defaultdict(int)
    for pid in ranked_hits:
        if pid not in substantive_ids:
            continue
        sec_key = (pages[pid]["document_id"], pages[pid]["section"])
        if section_counts[sec_key] >= HYBRID_MAX_SECTION_SEEDS:
            continue
        seeds_ids.append(pid)
        section_counts[sec_key] += 1
        if len(seeds_ids) >= HYBRID_SEED_K:
            break
    if len(seeds_ids) < HYBRID_SEED_K:
        for pid in ranked_hits:
            if pid in substantive_ids and pid not in seeds_ids:
                seeds_ids.append(pid)
                if len(seeds_ids) >= HYBRID_SEED_K:
                    break

    seed_set = set(seeds_ids)
    seeds = [
        {
            **pages[pid],
            "distance": distances[pid],
            "hops": 0,
            "path": [{"id": pid, "title": pages[pid]["title"]}],
        }
        for pid in seeds_ids
    ]

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in graph["nodes"]}
    same_section_adj = defaultdict(set)
    page_terms = defaultdict(set)
    term_pages = defaultdict(set)
    related_terms = defaultdict(set)

    for edge in graph["edges"]:
        a, b, rel = edge["source"], edge["target"], edge["relation"]
        if rel == "next_page" and a in pages and b in pages:
            if pages[a]["document_id"] == pages[b]["document_id"] and pages[a]["section"] == pages[b]["section"]:
                same_section_adj[a].add(b)
                same_section_adj[b].add(a)
        elif rel == "mentions":
            if a in pages and nodes.get(b, {}).get("kind") == "term":
                page_terms[a].add(b)
                term_pages[b].add(a)
            elif b in pages and nodes.get(a, {}).get("kind") == "term":
                page_terms[b].add(a)
                term_pages[a].add(b)
        elif rel == "related_term":
            if nodes.get(a, {}).get("kind") == "term" and nodes.get(b, {}).get("kind") == "term":
                related_terms[a].add(b)
                related_terms[b].add(a)

    # 2. チャネルA: 同一章内の前後1ページ（問題→解答、事故例→対策などの連続スライド補完）
    adj_candidates = {}
    for rank, seed_id in enumerate(seeds_ids):
        for nb in sorted(same_section_adj[seed_id]):
            if nb in seed_set or nb not in substantive_ids:
                continue
            dist = distances.get(nb, 1.0)
            if dist > 0.44 and lex_scores.get(nb, 0.0) < 0.15:
                continue
            forward_bonus = 0.05 if pages[nb]["page"] == pages[seed_id]["page"] + 1 else 0.02
            score = hybrid_rel(nb) + forward_bonus - 0.01 * rank
            cand = {
                **pages[nb],
                "distance": dist,
                "hops": 1,
                "channel": "same_section_adjacent",
                "path": [
                    {"id": seed_id, "title": pages[seed_id]["title"]},
                    {"id": nb, "title": pages[nb]["title"]},
                ],
            }
            if nb not in adj_candidates or score > adj_candidates[nb][0]:
                adj_candidates[nb] = (score, cand)

    adj_limit = max(0, min(HYBRID_MAX_ADJACENT, HYBRID_MAX_RESULTS - len(seeds)))
    sorted_adj = [cand for _, cand in sorted(adj_candidates.values(), key=lambda x: -x[0])]
    expanded_adjacent = []
    adj_sections = set()
    for cand in sorted_adj:
        sec_key = (cand["document_id"], cand["section"])
        if sec_key not in adj_sections and len(expanded_adjacent) < adj_limit:
            expanded_adjacent.append(cand)
            adj_sections.add(sec_key)
    for cand in sorted_adj:
        if cand not in expanded_adjacent and len(expanded_adjacent) < adj_limit:
            expanded_adjacent.append(cand)
    chosen_ids = seed_set | {c["page_id"] for c in expanded_adjacent}

    # 3. チャネルB: 特異度重み付き用語・関連用語ブリッジ（ハブ語除外＋クエリ類似度で再順位付け）
    def valid_term(tid: str) -> bool:
        deg = len(term_pages[tid])
        return 1 <= deg <= HYBRID_TERM_MAX_DEGREE

    term_candidates = {}
    seed_sections = {(pages[s]["document_id"], pages[s]["section"]) for s in seeds_ids}
    for seed_id in seeds_ids:
        for t1 in sorted(page_terms[seed_id]):
            if not valid_term(t1):
                continue
            idf1 = 1.0 / math.log2(len(term_pages[t1]) + 1.0)
            # 2-hop: page -> term -> page
            for target_pid in sorted(term_pages[t1]):
                if target_pid in chosen_ids or target_pid not in substantive_ids:
                    continue
                dist = distances.get(target_pid, 1.0)
                if dist > 0.38 and lex_scores.get(target_pid, 0.0) < 0.2:
                    continue
                cross_bonus = 0.04 if (pages[target_pid]["document_id"], pages[target_pid]["section"]) not in seed_sections else 0.0
                score = hybrid_rel(target_pid) + 0.15 * idf1 + cross_bonus
                cand = {
                    **pages[target_pid],
                    "distance": dist,
                    "hops": 2,
                    "channel": "term_bridge",
                    "path": [
                        {"id": seed_id, "title": pages[seed_id]["title"]},
                        {"id": t1, "title": nodes[t1]["title"]},
                        {"id": target_pid, "title": pages[target_pid]["title"]},
                    ],
                }
                if target_pid not in term_candidates or score > term_candidates[target_pid][0]:
                    term_candidates[target_pid] = (score, cand)
            # 3-hop: page -> term -> related_term -> page
            for t2 in sorted(related_terms[t1]):
                if not valid_term(t2):
                    continue
                idf2 = 1.0 / math.log2(len(term_pages[t2]) + 1.0)
                bridge_weight = 0.8 * min(idf1, idf2)
                for target_pid in sorted(term_pages[t2]):
                    if target_pid in chosen_ids or target_pid not in substantive_ids:
                        continue
                    dist = distances.get(target_pid, 1.0)
                    if dist > 0.36 and lex_scores.get(target_pid, 0.0) < 0.2:
                        continue
                    cross_bonus = 0.04 if (pages[target_pid]["document_id"], pages[target_pid]["section"]) not in seed_sections else 0.0
                    score = hybrid_rel(target_pid) + 0.15 * bridge_weight + cross_bonus
                    cand = {
                        **pages[target_pid],
                        "distance": dist,
                        "hops": 3,
                        "channel": "related_term_bridge",
                        "path": [
                            {"id": seed_id, "title": pages[seed_id]["title"]},
                            {"id": t1, "title": nodes[t1]["title"]},
                            {"id": t2, "title": nodes[t2]["title"]},
                            {"id": target_pid, "title": pages[target_pid]["title"]},
                        ],
                    }
                    if target_pid not in term_candidates or score > term_candidates[target_pid][0]:
                        term_candidates[target_pid] = (score, cand)

    remaining_slots = max(0, HYBRID_MAX_RESULTS - len(seeds) - len(expanded_adjacent))
    expanded_terms = [
        cand for _, cand in sorted(term_candidates.values(), key=lambda x: -x[0])
    ][:remaining_slots]

    return {
        "seeds": seeds,
        "expanded_adjacent": expanded_adjacent,
        "expanded_terms": expanded_terms,
    }



def run_full_context(agent, question: str, model, instructions: str, prompt_template: str):
    """全資料を一括投入し、検索ツールを使わずに回答する。"""
    prompt = prompt_template.format(materials=get_all_materials(), question=question)
    print(f"[full-context] 入力{len(prompt):,}文字")
    try:
        with agent.override(tools=[], instructions=instructions):
            return agent.run_sync(prompt, model=model)
    except ModelHTTPError as e:
        print(f"[full-context] APIエラー HTTP {e.status_code}: {e.body}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="エージェントを使わず、講義資料の検索だけを実行する")
    parser.add_argument("query", help="調べたい内容")
    args = parser.parse_args()
    for i, result in enumerate(search(args.query), 1):
        print(f"\n{'=' * 60}\n検索結果 {i}\n{'=' * 60}")
        print(f"講義: {result['document_title']}")
        print(f"章: {result['section']}")
        print(f"タイトル: {result['title']}")
        print(f"ページ: {result['page']}")
        print(f"\n{result['content']}")
