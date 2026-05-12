#!/usr/bin/env python3

import argparse
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Append a pipeline step to a markdown log.")
    parser.add_argument("--doc", required=True, help="Markdown file to append to")
    parser.add_argument("--title", required=True, help="Section title")
    parser.add_argument("--goal", required=True, help="What the step does")
    parser.add_argument("--script", action="append", default=[], help="Absolute script path")
    parser.add_argument("--command", action="append", default=[], help="Exact command used")
    parser.add_argument("--inputs", action="append", default=[], help="Input path")
    parser.add_argument("--outputs", action="append", default=[], help="Output path")
    parser.add_argument("--notes", action="append", default=[], help="Additional note")
    args = parser.parse_args()

    doc = Path(args.doc)
    doc.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("")
    lines.append(f"### {args.title}")
    lines.append("")
    lines.append(f"- Time: `{timestamp}`")
    lines.append(f"- Goal: {args.goal}")

    if args.script:
        lines.append("- Script(s):")
        for item in args.script:
            lines.append(f"  - `{item}`")

    if args.command:
        lines.append("- Command(s):")
        for item in args.command:
            lines.append(f"```bash\n{item}\n```")

    if args.inputs:
        lines.append("- Input(s):")
        for item in args.inputs:
            lines.append(f"  - `{item}`")

    if args.outputs:
        lines.append("- Output(s):")
        for item in args.outputs:
            lines.append(f"  - `{item}`")

    if args.notes:
        lines.append("- Note(s):")
        for item in args.notes:
            lines.append(f"  - {item}")

    with doc.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
