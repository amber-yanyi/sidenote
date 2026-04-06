"""
Sidenote — 核心逻辑模块
文章提取、LLM处理、HTML渲染
"""

import json
import os
import sys

import trafilatura
from google import genai


def extract_article(url: str) -> tuple[str, str]:
    """从URL提取文章标题和正文"""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"无法获取 {url}")

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    metadata = trafilatura.extract(downloaded, output_format="json")
    title = ""
    if metadata:
        meta = json.loads(metadata)
        title = meta.get("title", "")

    if not text:
        raise ValueError("无法提取文章内容")

    return title, text


def process_with_llm(title: str, text: str) -> dict:
    """调用Gemini API进行翻译和批注"""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("请设置 GEMINI_API_KEY 环境变量")
    client = genai.Client(api_key=api_key)

    prompt = f"""你是一位资深科技行业分析师，同时也是出色的中英翻译。现在请你帮一位同样资深但时间有限的中文读者处理一篇英文文章。

任务：
1. 逐段翻译成中文（不是摘要，是完整翻译）
2. 在少数关键位置加旁批注释（用你的外部知识补充背景）
3. 在最后给出总结：核心观点 + why this matters

翻译要求：
- 保留关键术语英文（如 MCP、CLI、agent、LLM 等）
- 翻译风格自然流畅，信息密度高，不要机翻腔
- 如果原文本身就是中文，则"翻译"栏直接保留原文

批注要求（最重要，请严格遵守）：
- 绝大多数段落不需要注释。一篇文章通常只需要 2-4 条批注
- 不要注释人名。如果一个人重要，文章自己会介绍。不要做人物百科
- 不要分析作者的遣词造句、修辞手法或写作意图

只在以下四种情况加批注：
1. 行业黑话与缩写：离开特定语境难以理解的前沿技术词汇（如 MCP、RAG、MoE），用一两句大白话解释它是什么、为什么重要
2. 特定实体与新事物：具有时效性的新模型、特定论文、新兴初创公司，补充读者可能不知道的背景
3. 隐式上下文与行业梗：作者一笔带过但影响深度理解的行业事件、历史背景或争议，把省略的 context 补全
4. 反直觉/复杂论断：涉及底层技术逻辑、需要简单拆解才能看懂的核心结论，用大白话拆解一下

- 每条注释 2-3 句话，语气像一个消息灵通的朋友在旁边补充 context

总结要求（非常重要，请认真对待）：
你的总结不是复述文章内容。你要做的是：像一个顶级行业分析师一样，穿透文章表面，揭示背后真正的逻辑。

具体结构：
1. 一句话穿透本质：用"不是 X，而是 Y"的框架，点破这篇文章真正在说什么。例如"OpenAI 不是在买媒体，而是在买 narrative infrastructure"。
2. 真正的战略原因：文章表面说的理由背后，真正的驱动力是什么？为什么是现在？
3. 行业级 insight：从这个个案拉升到行业趋势。这不只是关于这一家公司，而是关于什么更大的变化？
4. 读者可带走的判断：如果读者是创业者/从业者/投资人，这篇文章改变了他的哪个认知？他应该据此做什么？

风格要求：
- 直接、犀利、信息密度高
- 用短句，不要写长段落
- 敢下判断，不要两边都说
- 像一个见过世面的朋友在跟你说"这件事真正的 point 是..."

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
          "note": "2-3句话，补充外部背景知识"
        }}
      ]
    }}
  ],
  "summary": {{
    "one_liner": "一句话穿透本质，用'不是X，而是Y'的框架",
    "real_reason": "2-3句话，揭示表面叙事背后真正的战略逻辑",
    "industry_insight": "2-3句话，从个案拉到行业级趋势",
    "takeaway": "1-2句话，读者可以带走的判断或行动建议"
  }}
}}

annotations 数组可以为空（大多数段落应该为空）。整篇文章的批注总数控制在 2-4 条。

---

文章标题：{title}

文章正文：
{text}"""

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
        raise ValueError(f"Gemini返回的JSON格式不正确: {e}")


def _render_summary(summary) -> str:
    """渲染总结区块，兼容字符串和结构化格式"""
    if not summary:
        return ""
    if isinstance(summary, str):
        return f'<div class="summary"><h2>Why This Matters</h2><div class="summary-section"><p>{summary}</p></div></div>'
    # 结构化格式
    one_liner = summary.get("one_liner", "")
    real_reason = summary.get("real_reason", "")
    industry_insight = summary.get("industry_insight", "")
    takeaway = summary.get("takeaway", "")
    html = '<div class="summary"><h2>Why This Matters</h2>'
    if one_liner:
        html += f'<div class="one-liner">{one_liner}</div>'
    if real_reason:
        html += f'<div class="summary-section"><h3>Real Reason</h3><p>{real_reason}</p></div>'
    if industry_insight:
        html += f'<div class="summary-section"><h3>Industry Insight</h3><p>{industry_insight}</p></div>'
    if takeaway:
        html += f'<div class="summary-section"><h3>Takeaway</h3><p>{takeaway}</p></div>'
    html += '</div>'
    return html


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

    .summary {{
        margin-top: 48px;
        padding: 32px;
        background: #f5f3ee;
        border-radius: 12px;
        border: 1px solid #e0ddd5;
    }}

    .summary h2 {{
        font-size: 20px;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 24px;
    }}

    .summary .one-liner {{
        font-size: 18px;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 24px;
        line-height: 1.6;
    }}

    .summary-section {{
        margin-bottom: 20px;
    }}

    .summary-section:last-child {{
        margin-bottom: 0;
    }}

    .summary-section h3 {{
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #999;
        margin-bottom: 8px;
    }}

    .summary-section p {{
        font-size: 16px;
        color: #333;
        line-height: 1.8;
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
    {_render_summary(data.get("summary"))}

</div>
</body>
</html>"""
