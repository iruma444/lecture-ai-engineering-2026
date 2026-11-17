"""第5回宿題（発展）: 構造認識型ハイブリッド検索と改良目次探索の単体テスト（APIキー不要）。"""

import json

from pydantic_ai.models.test import TestModel

from app import retrieval, tools
from app.agent import HYBRID_INSTRUCTIONS, agent
from scripts.build_rag_index import build_index


def test_browse_outline_removes_redundant_headers_and_returns_full_page():
    sec = retrieval.browse_outline("lec02-main-s045")
    by_id = {c["id"]: c for c in sec["children"]}
    # 既存 browse_tree では "4/4 　選んだモデルを、いつ・どう直すか" になっていたページで実質要約文が返る
    assert "元の重みは固定" in by_id["lec02-main-p052"]["content"]
    assert by_id["lec02-main-p052"]["content"] != "4/4 　選んだモデルを、いつ・どう直すか"

    sec3 = retrieval.browse_outline("lec03-main-s037")
    by_id3 = {c["id"]: c for c in sec3["children"]}
    # 既存 browse_tree ではタイトル重複 ("実装例：OpenClaw") になっていたページで本文先頭文が返る
    assert "長期メモリ" in by_id3["lec03-main-p040"]["content"]

    leaf = retrieval.browse_outline("lec02-main-p052")
    assert leaf["children"] == []
    assert "QLoRA" in leaf["node"]["content"]


def test_is_substantive_page_filters_section_cover_slides():
    pages = {p["page_id"]: p for p in retrieval.load_pages()}
    assert not retrieval.is_substantive_page(pages["lec03-main-p037"])  # "Part 6 メモリ" 章扉
    assert retrieval.is_substantive_page(pages["lec03-main-p039"])
    assert retrieval.is_substantive_page(pages["lec02-main-p052"])


def test_search_hybrid_combines_same_section_adjacent_and_term_bridges(tmp_path, monkeypatch):
    source = tmp_path / "lectures.json"
    graph_path = tmp_path / "graph.json"
    database = tmp_path / "index.sqlite3"

    docs = [
        {
            "id": "lec02-main",
            "title": "基盤モデルの仕組み",
            "content": [
                {
                    "section": "章A",
                    "content": [
                        {
                            "id": "lec02-main-p001",
                            "title": "クイズ問題",
                            "page": 1,
                            "content": "クイズ問題\n1日1万件の問い合わせ振り分けで最初に試すべき選択肢はどれか説明せよ。",
                        },
                        {
                            "id": "lec02-main-p002",
                            "title": "クイズ解答",
                            "page": 2,
                            "content": "クイズ解答\n答えは小さな特化モデルで評価セットにより精度とコストを確かめることである。",
                        },
                    ],
                },
                {
                    "section": "章B",
                    "content": [
                        {
                            "id": "lec02-main-p003",
                            "title": "章B扉",
                            "page": 3,
                            "content": "Part 2\n章B扉",
                        },
                        {
                            "id": "lec02-main-p004",
                            "title": "無関係な異章ページ",
                            "page": 4,
                            "content": "無関係な異章ページ\n画像生成と拡散モデルの説明であり問い合わせ振り分けとは無関係である。",
                        },
                    ],
                },
            ],
        },
        {
            "id": "lec03-main",
            "title": "AIエージェント",
            "content": [
                {
                    "section": "Appendix",
                    "content": [
                        {
                            "id": "lec03-main-p010",
                            "title": "関連用語ブリッジ先",
                            "page": 10,
                            "content": "関連用語ブリッジ先\n評価セットと分類器を組み合わせた最新運用の補足説明である。",
                        },
                        {
                            "id": "lec03-main-p011",
                            "title": "同音異義ノイズページ",
                            "page": 11,
                            "content": "同音異義ノイズページ\n全く別の話題でクエリとのベクトル類似度が低いページである。",
                        },
                    ],
                }
            ],
        },
    ]
    source.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")

    graph = {
        "nodes": [
            {"id": "lec02-main-p001", "kind": "page", "title": "クイズ問題"},
            {"id": "lec02-main-p002", "kind": "page", "title": "クイズ解答"},
            {"id": "lec02-main-p003", "kind": "page", "title": "章B扉"},
            {"id": "lec02-main-p004", "kind": "page", "title": "無関係な異章ページ"},
            {"id": "lec03-main-p010", "kind": "page", "title": "関連用語ブリッジ先"},
            {"id": "lec03-main-p011", "kind": "page", "title": "同音異義ノイズページ"},
            {"id": "term:eval-set", "kind": "term", "title": "評価セット"},
            {"id": "term:rubric", "kind": "term", "title": "ルーブリック"},
            {"id": "term:homonym", "kind": "term", "title": "同音語"},
        ],
        "edges": [
            {"source": "lec02-main-p001", "target": "lec02-main-p002", "relation": "next_page"},
            {"source": "lec02-main-p002", "target": "lec02-main-p003", "relation": "next_page"},
            {"source": "lec02-main-p003", "target": "lec02-main-p004", "relation": "next_page"},
            {"source": "lec02-main-p001", "target": "term:eval-set", "relation": "mentions"},
            {"source": "term:eval-set", "target": "term:rubric", "relation": "related_term"},
            {"source": "lec03-main-p010", "target": "term:rubric", "relation": "mentions"},
            {"source": "lec02-main-p001", "target": "term:homonym", "relation": "mentions"},
            {"source": "lec03-main-p011", "target": "term:homonym", "relation": "mentions"},
        ],
    }
    graph_path.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")

    vec_map = {
        "クイズ問題": [1.0, 0.0],
        "クイズ解答": [0.92, 0.39],
        "章B扉": [0.90, 0.43],
        "無関係な異章ページ": [0.0, 1.0],
        "関連用語ブリッジ先": [0.94, 0.34],
        "同音異義ノイズページ": [0.0, 1.0],
    }

    def embed_doc(text, _task):
        for key, vec in vec_map.items():
            if key in text:
                return vec
        return [0.0, 1.0]

    build_index(source, database, embedder=embed_doc)
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: [1.0, 0.0])
    monkeypatch.setattr(retrieval, "HYBRID_SEED_K", 1)
    monkeypatch.setattr(retrieval, "HYBRID_MAX_ADJACENT", 1)
    monkeypatch.setattr(retrieval, "HYBRID_MAX_RESULTS", 3)

    res = retrieval.search_hybrid("問い合わせ振り分け", source=source, graph_path=graph_path, db_path=database)
    assert [s["page_id"] for s in res["seeds"]] == ["lec02-main-p001"]
    assert [a["page_id"] for a in res["expanded_adjacent"]] == ["lec02-main-p002"]
    assert [t["page_id"] for t in res["expanded_terms"]] == ["lec03-main-p010"]


def test_hybrid_agent_tools_wiring(monkeypatch):
    monkeypatch.setattr(
        "app.retrieval.search_hybrid",
        lambda *args, **kwargs: {"seeds": [{"page_id": "lec02-main-p052"}], "expanded_adjacent": [], "expanded_terms": []},
    )
    model = TestModel(call_tools=["search_hybrid"])
    with agent.override(instructions=HYBRID_INSTRUCTIONS, tools=[tools.search_hybrid, tools.browse_outline], model=model):
        result = agent.run_sync("LoRAとは？")
    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert offered == {"search_hybrid", "browse_outline"}
    assert result.output


def test_lexical_idf_boost_rescues_rare_proper_noun_page():
    pages = {p["page_id"]: p for p in retrieval.load_pages()}
    scores = retrieval._lexical_idf_scores("RSI Google DeepMind AI開発", pages)
    assert scores["lec01-main-p011"] > scores["lec03-main-p055"]
