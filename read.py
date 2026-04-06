#!/usr/bin/env python3
"""
Read — AI文章阅读增强工具
输入文章URL或文本，输出带中文翻译和行内批注的HTML页面。
"""

import argparse
import json
import os
import sys
import tempfile
import webbrowser
from pathlib import Path

import trafilatura
from google import genai


def extract_article(url: str) -> tuple[str, str]:
    """从URL提取文章标题和正文"""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print(f"错误：无法获取 {url}", file=sys.stderr)
        sys.exit(1)

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    metadata = trafilatura.extract(downloaded, output_format="json")
    title = ""
    if metadata:
        meta = json.loads(metadata)
        title = meta.get("title", "")

    if not text:
        print("错误：无法提取文章内容", file=sys.stderr)
        sys.exit(1)

    return title, text


def process_with_llm(title: str, text: str) -> dict:
    """调用Gemini API进行翻译和批注"""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("错误：请设置 GEMINI_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    prompt = f"""你是一位资深AI行业分析师，同时也是出色的中英翻译。现在请你帮一位同样资深但时间有限的中文读者处理一篇英文文章。

任务：
1. 逐段翻译成中文（不是摘要，是完整翻译）
2. 在值得关注的地方加旁批注释

翻译要求：
- 保留关键术语英文（如 MCP、CLI、agent、LLM 等）
- 翻译风格自然流畅，信息密度高，不要机翻腔
- 如果原文本身就是中文，则"翻译"栏直接保留原文

批注要求（最重要）：
- 不是每段都需要注释，只标注真正有信息增量的地方
- 重点关注：反直觉的判断、行业趋势/转折信号、隐含的因果关系、需要背景知识才能理解的点
- 每条注释 2-3 句话，解释"为什么这很重要"或"背后的 context 是什么"
- 语气像一个懂行的朋友在旁边小声点评

请严格以如下JSON格式输出，不要输出其他内容：

{{
  "title": "原文标题",
  "title_zh": "中文标题",
  "paragraphs": [
    {{
      "original": "原文段落",
      "translation": "中文翻译",
      "annotations": [
        {{
          "sentence": "被注释的那句话的中文翻译（不是英文原文）",
          "note": "2-3句话的旁批"
        }}
      ]
    }}
  ]
}}

annotations 数组可以为空（大多数段落不需要注释）。

---

文章标题：{title}

文章正文：
{text}"""

    print("正在处理文章...", file=sys.stderr)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    content = response.text

    # 尝试提取JSON（处理可能的markdown代码块包裹）
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"错误：Gemini返回的JSON格式不正确: {e}", file=sys.stderr)
        print(f"原始返回:\n{content[:500]}", file=sys.stderr)
        sys.exit(1)


def render_html(data: dict, url: str = "") -> str:
    """将处理结果渲染为HTML"""
    title = data.get("title", "")
    title_zh = data.get("title_zh", "")
    paragraphs = data.get("paragraphs", [])

    rows_html = ""
    for p in paragraphs:
        original = p["original"].replace("\n", "<br>")
        translation = p["translation"].replace("\n", "<br>")

        annotations_html = ""
        for ann in p.get("annotations", []):
            sentence = ann["sentence"]
            note = ann["note"]
            annotations_html += f"""
            <div class="annotation">
                <div class="annotation-quote">&ldquo;{sentence}&rdquo;</div>
                <div class="annotation-note">{note}</div>
            </div>"""

        rows_html += f"""
        <div class="row">
            <div class="col original">{original}</div>
            <div class="col translation">
                {translation}
                {annotations_html}
            </div>
        </div>"""

    source_line = f'<div class="source">原文：<a href="{url}" target="_blank">{url}</a></div>' if url else ""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_zh or title}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
                     "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
        background: #fafaf7;
        color: #333;
        line-height: 1.8;
    }}

    .container {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 40px 20px;
    }}

    header {{
        text-align: center;
        margin-bottom: 48px;
        padding-bottom: 32px;
        border-bottom: 1px solid #e0ddd5;
    }}

    h1 {{
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #1a1a1a;
    }}

    .subtitle {{
        font-size: 18px;
        color: #666;
    }}

    .source {{
        margin-top: 12px;
        font-size: 13px;
        color: #999;
    }}

    .source a {{
        color: #999;
        text-decoration: underline;
    }}

    .columns-header {{
        display: flex;
        gap: 40px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e0ddd5;
    }}

    .columns-header span {{
        flex: 1;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #999;
    }}

    .row {{
        display: flex;
        gap: 40px;
        padding: 20px 0;
        border-bottom: 1px solid #eee;
    }}

    .row:last-child {{
        border-bottom: none;
    }}

    .col {{
        flex: 1;
        font-size: 16px;
    }}

    .original {{
        color: #666;
    }}

    .translation {{
        color: #1a1a1a;
    }}

    .annotation {{
        margin-top: 14px;
        padding: 12px 16px;
        background: #fef9ef;
        border-left: 3px solid #e8a735;
        border-radius: 0 6px 6px 0;
        font-size: 14px;
    }}

    .annotation-quote {{
        color: #b08520;
        font-style: italic;
        margin-bottom: 6px;
        font-size: 13px;
    }}

    .annotation-note {{
        color: #5a4a20;
        line-height: 1.7;
    }}

    @media (max-width: 768px) {{
        .row {{
            flex-direction: column;
            gap: 16px;
        }}
        .columns-header {{
            display: none;
        }}
        .original {{
            padding-bottom: 12px;
            border-bottom: 1px dashed #ddd;
        }}
    }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>{title_zh or title}</h1>
        {"<div class='subtitle'>" + title + "</div>" if title_zh and title else ""}
        {source_line}
    </header>
    <div class="columns-header">
        <span>Original</span>
        <span>中文翻译 &amp; 批注</span>
    </div>
    {rows_html}
</div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Read — AI文章阅读增强工具")
    parser.add_argument("input", help="文章URL或本地文件路径")
    parser.add_argument("-o", "--output", help="输出HTML文件路径（默认自动打开临时文件）")
    parser.add_argument("--no-open", action="store_true", help="不自动在浏览器中打开")
    args = parser.parse_args()

    # 判断输入是URL还是文件
    url = ""
    if args.input.startswith(("http://", "https://")):
        url = args.input
        title, text = extract_article(url)
        print(f"已提取文章：{title}", file=sys.stderr)
    else:
        # 当作本地文件读取
        path = Path(args.input)
        if not path.exists():
            print(f"错误：文件不存在 {args.input}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8")
        title = path.stem

    # 调用Claude处理
    data = process_with_llm(title, text)

    # 渲染HTML
    html = render_html(data, url)

    # 输出
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(html, encoding="utf-8")
        print(f"已保存到 {out_path}", file=sys.stderr)
        if not args.no_open:
            webbrowser.open(f"file://{out_path.resolve()}")
    else:
        with tempfile.NamedTemporaryFile(
            suffix=".html", prefix="read_", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(html)
            tmp_path = f.name
        print(f"已生成 {tmp_path}", file=sys.stderr)
        if not args.no_open:
            webbrowser.open(f"file://{tmp_path}")


if __name__ == "__main__":
    main()
