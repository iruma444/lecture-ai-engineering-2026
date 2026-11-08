# 第3回 解答

## 演習

- 第1回（素のLLM）とエージェントの回答の違い：
  - **第1回（素のLLM `app.ask`）**：講座固有の知識を持たないため、「宿題の3レベル」に対して「レベル1：基礎（初級）、レベル2：応用（中級）、レベル3：発展（上級）」のように一般論から推測した誤答（ハルシネーション）を生成した。
  - **第3回（エージェント `app.agent`）**：回答前に `search_glossary` ツールを呼び出して `data/glossary.json` を検索し、「1. 基礎（講義内容の確認）、2. 実装（自分でコードを書く）、3. 発展（調査や試行）の3段階があり、どれか1つに取り組めば完了（原則として採点はしない）」と用語集の定義通りに正確に回答した。また、`INSTRUCTIONS` の指示に従い末尾に「（関連回：第1回）」を明示した。
- `--trace` で見えたツール呼び出し（ツール名と引数）：
  - 質問 `"LoRAとQLoRAの違いは？"` の実行時：
    1. `[tool call] search_glossary({'query': 'LoRA'})`
    2. `[tool call] search_glossary({'query': 'QLoRA'})`
    - `search_glossary` の docstring に「`query` には調べたい用語を1つだけ入れる」と記載されているため、1回の応答で2つのツール呼び出し（バッチコール）が発行された。
  - 質問 `"第2回の内容から確認問題を1問出して"` の実行時：
    1. `[tool call] make_quiz({'lecture': 2})`
- 1つの質問での LLM 呼び出し回数（`requests`）：
  - `"LoRAとQLoRAの違いは？"`：`requests=2`（`tool_calls=2`, `input_tokens=1196`, `output_tokens=196`）
  - `"第2回の内容から確認問題を1問出して"`：`requests=2`（`tool_calls=1`, `input_tokens=914`, `output_tokens=69`）
  - いずれも「①ユーザーの質問を受け取りツール呼び出しを決定する1回目」と「②ツールの実行結果を受け取り最終回答を生成する2回目」の計2回 LLM が呼ばれている。
- 追加したツールと、docstring に書いた説明：
  - **追加したツール**：`make_quiz(lecture: int) -> str`（`app/tools.py` に実装し、`app/agent.py` で `agent.tool_plain(tools.make_quiz)` として登録）
  - **docstring に書いた説明**：
    ```text
    指定した回（1始まりの整数）の用語から1つを選び、確認問題と答えを返す。

    「第2回の確認問題を出して」「復習クイズを出して」のように、
    指定した回の復習用の問題を求められたときに使う。
    ```

## 宿題

- 取り組んだレベル：発展（発展 a：Google ADK でエージェントを組み立て直し、Pydantic AI と書き方や回答を比較する）
- やったこと：
  - `app/agent.py` の `INSTRUCTIONS` と `app/tools.py` の3つのツール関数（`search_glossary`, `get_lecture_summary`, `make_quiz`）をそのまま流用し、`adk_agent/glossary_agent/agent.py` および `adk_agent/glossary_agent/__init__.py` を新規作成した。
  - `PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent` を実行し、演習と同じ3つの質問（①宿題の3レベル、②LoRAとQLoRAの違い、③第2回の確認問題の出題）を投げて、フレームワークの書き方とエージェントの挙動・回答を比較した。
- 結果：
  - **1. 書き方（指示とツールの渡し方）の比較**：

    | 比較観点 | Pydantic AI (`app/agent.py`) | Google ADK (`adk_agent/glossary_agent/agent.py`) |
    | :--- | :--- | :--- |
    | **クラスとモジュール** | `from pydantic_ai import Agent` | `from google.adk.agents import Agent` |
    | **システムプロンプトの引数名** | `instructions=INSTRUCTIONS`（複数形） | `instruction=INSTRUCTIONS`（単数形） |
    | **モデルの指定方法** | 初期化時または `agent.run_sync(..., model=build_model())` で実行時に注入可能（テスト時は `TestModel` に差し替え） | `Agent(model="gemini-3.5-flash-lite", name="glossary_agent", ...)` でエージェント名（`name`）とともに定義 |
    | **ツールの登録方法** | `@agent.tool_plain` デコレータや `agent.tool_plain(fn)` で登録（または `tools=[...]`） | `tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz]` のように通常のPython関数のリストをそのまま渡す |
    | **ツールのスキーマ生成** | 関数の型ヒントと docstring から自動生成 | 同じく関数の型ヒントと docstring から自動生成（`tools.py` を一切変更せずそのまま流用できた） |
    | **実行・対話の仕組み** | `run_sync()` 等を自前の `main()` で呼び出し、CLI引数（`--trace` 等）を自作する | `root_agent` 変数を定義しておけば、`adk run`（対話型CLI）や `adk web`（ブラウザUI）がフレームワーク標準で提供される |

  - **2. 同じ質問に対する回答の比較**：
    - **質問①「AIE2026の講座の宿題にある『3レベル』とは何ですか？」**：ADKでも `search_glossary` が呼ばれ、「1. 基礎：講義内容の確認、2. 実装：自分でコードを書く、3. 発展：調査や試行」の3段階でどれか1つで完了・原則採点なしという点、末尾の「関連する回：第1回」まで Pydantic AI と同等に正確な回答が返った。
    - **質問②「LoRAとQLoRAの違いは？」**：ADKでも両用語を検索した上で、LoRA（重みを固定し低ランク行列のみ学習するPEFT手法）とQLoRA（重みを4ビット等に量子化してメモリを節約した上でLoRA学習する手法）の違いおよび「関連する回：第2回」が正確に回答された。
    - **質問③「第2回の内容から確認問題を1問出して」**：ADKでも追加した `make_quiz(lecture=2)` が呼び出された。ツールは問題と答え（`answer`）の両方をJSONで返しているが、`INSTRUCTIONS` の「宿題の答えそのものは書かず、考え方のヒントで導く」という指示が効き、Pydantic AI（出題：「Hugging Face Hub」）・ADK（出題：「PEFT」）ともに**最初から答えをネタバレせず、まず問題文だけをユーザーに出題して考えさせる対話的な振る舞い**になった。
- 気づき・疑問：
  - **ツール関数のポータビリティの高さ**：`app/tools.py` の関数は特定のフレームワークに依存しない純粋なPython関数（型ヒント＋docstring）として書かれているため、コードを1行も書き換えることなく Pydantic AI と Google ADK の両方でそのままツールとして認識・実行できた。フレームワークが変わっても「モデル＋ループ＋ツール」の本質的な構造は同じであり、モデルの挙動を支配しているのはフレームワークの差ではなく **`INSTRUCTIONS`（システムプロンプト）と関数の `docstring`・型定義** であることが実感できた。
  - **フレームワークごとの設計思想の違い**：Pydantic AI は `TestModel` によるAPI不要の単体テスト（`pytest`）やプログラムへの組み込みがしやすいコード中心の設計であるのに対し、Google ADK は `root_agent` を定義するだけで `adk run` によるマルチターン対話ループや `/tmp/agents_log/` へのロギングが即座に立ち上がるなど、ハーネス機能が最初から厚く同梱されている違いが分かった。
