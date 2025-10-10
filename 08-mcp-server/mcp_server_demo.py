#!/usr/bin/env python3
"""
Module 6: Minimal MCP-like Server Demo

这是一个最小可运行的 MCP 风格（行分隔 JSON-RPC）的本地服务器示例：
- 通过 stdin/stdout 与客户端通信（易于跨平台）
- 支持列出工具（list_tools）和调用工具（call_tool）两类方法
- 工具示例：get_time、echo、add

注意：这是教学用的“最小实现”，不依赖外部三方库，帮助理解 MCP 的交互流程。
"""
import sys
import json
import datetime
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_time",
        "description": "Return current date and time in ISO format",
        "args_schema": [],
    },
    {
        "name": "echo",
        "description": "Echo back the provided message",
        "args_schema": [
            {"name": "message", "type": "string", "required": True, "description": "Text to echo"},
        ],
    },
    {
        "name": "add",
        "description": "Add two numbers (a + b)",
        "args_schema": [
            {"name": "a", "type": "number", "required": True},
            {"name": "b", "type": "number", "required": True},
        ],
    },
]


def handle_list_tools() -> Dict[str, Any]:
    return {"tools": TOOLS}


def handle_call_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    args = params.get("args", {})

    if name == "get_time":
        now = datetime.datetime.now().isoformat()
        return {"output": now}

    if name == "echo":
        message = str(args.get("message", ""))
        return {"output": message}

    if name == "add":
        try:
            a = float(args.get("a", 0))
            b = float(args.get("b", 0))
        except (TypeError, ValueError):
            return {"error": "Invalid arguments: 'a' and 'b' must be numbers"}
        return {"output": a + b}

    return {"error": f"Unknown tool: {name}"}


def process_request(line: str) -> Optional[Dict[str, Any]]:
    try:
        req = json.loads(line)
    except json.JSONDecodeError:
        return {"id": None, "error": "Invalid JSON"}

    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "list_tools":
        result = handle_list_tools()
        return {"id": req_id, "result": result}

    if method == "call_tool":
        result = handle_call_tool(params)
        return {"id": req_id, "result": result}

    return {"id": req_id, "error": f"Unknown method: {method}"}


def main() -> None:
    try:
        print(json.dumps({"ready": True}), flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            resp = process_request(line)
            if resp is not None:
                print(json.dumps(resp), flush=True)
    except KeyboardInterrupt:
        # 优雅退出，不抛出异常回溯
        pass


if __name__ == "__main__":
    main()