#!/usr/bin/env python3
"""
Module 6: Minimal MCP-like Client Demo

与本地最小 MCP 风格服务器（mcp_server_demo.py）进行交互：
- 通过 stdin/stdout 启动子进程并进行行分隔 JSON 通信
- 提供 list_tools 和 call_tool 的交互演示

用法：
1) 先启动服务端：
   python /Users/mac/Documents/ai/langchain-learn/langchain-learn/06-mcp/mcp_server_demo.py
   或由客户端自动启动子进程

2) 再运行客户端：
   python /Users/mac/Documents/ai/langchain-learn/langchain-learn/06-mcp/mcp_client_demo.py
"""
import os
import sys
import json
import time
import subprocess
from typing import Any, Dict, Optional

sys.stdout.reconfigure(encoding="utf-8")

SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server_demo.py")

class MCPClient:
    def __init__(self, server_path: str) -> None:
        self.server_path = server_path
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        # 等待服务端 ready
        assert self.proc.stdout is not None
        ready_line = self.proc.stdout.readline().strip()
        if ready_line:
            try:
                data = json.loads(ready_line)
                if not data.get("ready"):
                    raise RuntimeError("Server not ready")
            except Exception as e:
                raise RuntimeError(f"Failed to read server ready state: {e}")

    def send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        assert self.proc and self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        return json.loads(line)

    def list_tools(self) -> Dict[str, Any]:
        return self.send({"id": int(time.time()), "method": "list_tools", "params": {}})

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return self.send({"id": int(time.time()), "method": "call_tool", "params": {"name": name, "args": args}})

    def stop(self) -> None:
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


def main() -> None:
    client = MCPClient(SERVER_PATH)
    client.start()

    print("===== Minimal MCP-like Client Demo =====")
    # 列出工具
    resp = client.list_tools()
    tools = resp.get("result", {}).get("tools", [])
    print("可用工具:")
    for t in tools:
        print(f"- {t['name']}: {t.get('description', '')}")

    # 调用工具演示
    print("\n调用 get_time:")
    print(client.call_tool("get_time", {}))

    print("\n调用 echo(message='Hello MCP'):")
    print(client.call_tool("echo", {"message": "Hello MCP"}))

    print("\n调用 add(a=15, b=23):")
    print(client.call_tool("add", {"a": 15, "b": 23}))

    client.stop()

if __name__ == "__main__":
    main()