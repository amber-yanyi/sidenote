#!/usr/bin/env python3
"""
Sidenote CLI — AI文章阅读增强工具
输入文章URL或文本，输出带中文翻译和行内批注的HTML页面。
"""

import argparse
import sys
import tempfile
import webbrowser
from pathlib import Path

from core import extract_article, process_with_llm, render_html


def main():
    parser = argparse.ArgumentParser(description="Sidenote — AI文章阅读增强工具")
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
        path = Path(args.input)
        if not path.exists():
            print(f"错误：文件不存在 {args.input}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8")
        title = path.stem

    print("正在处理文章...", file=sys.stderr)
    data = process_with_llm(title, text)

    html = render_html(data, url)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(html, encoding="utf-8")
        print(f"已保存到 {out_path}", file=sys.stderr)
        if not args.no_open:
            webbrowser.open(f"file://{out_path.resolve()}")
    else:
        with tempfile.NamedTemporaryFile(
            suffix=".html", prefix="sidenote_", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write(html)
            tmp_path = f.name
        print(f"已生成 {tmp_path}", file=sys.stderr)
        if not args.no_open:
            webbrowser.open(f"file://{tmp_path}")


if __name__ == "__main__":
    main()
