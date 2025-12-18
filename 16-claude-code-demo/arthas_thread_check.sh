#!/bin/bash

# Arthas 线程排查脚本
# 使用方法: ./arthas_thread_check.sh <PID>

set -e

if [ -z "$1" ]; then
    echo "❌ 缺少进程 PID"
    echo "用法: $0 <PID>"
    echo ""
    echo "查找 Java 进程:"
    jps -l
    exit 1
fi

PID=$1
OUTPUT_DIR="./thread-analysis-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "🔍 开始排查进程: $PID"
echo "📂 输出目录: $OUTPUT_DIR"
echo ""

# 检查 Arthas 是否已安装
if ! command -v as.sh &> /dev/null; then
    echo "⚠️  Arthas 未安装，正在下载..."
    curl -L https://arthas.aliyun.com/install.sh | sh
fi

# 创建 Arthas 命令文件
cat > "$OUTPUT_DIR/commands.txt" <<EOF
# 1. 线程概览
dashboard -n 1

# 2. 导出所有线程
thread --all

# 3. 查看最忙的5个线程
thread -n 5

# 4. 检查死锁
thread -b

# 5. 查看 WAITING 线程
thread --state WAITING

# 6. 查看 TIMED_WAITING 线程
thread --state TIMED_WAITING

# 7. 查看 BLOCKED 线程
thread --state BLOCKED

# 8. JVM 信息
jvm

# 9. 内存使用
memory

# 10. 退出
quit
EOF

echo "📋 执行 Arthas 诊断命令..."
echo ""

# 使用 Arthas 批量执行命令
as.sh $PID < "$OUTPUT_DIR/commands.txt" > "$OUTPUT_DIR/arthas-output.txt" 2>&1

echo "✅ 诊断完成!"
echo ""

# 分析结果
echo "=" | awk '{for(i=1;i<=80;i++)printf "="; printf "\n"}'
echo "【分析结果】"
echo "=" | awk '{for(i=1;i<=80;i++)printf "="; printf "\n"}'

# 统计线程状态
echo ""
echo "1. 线程状态统计:"
grep -E "RUNNABLE|WAITING|TIMED_WAITING|BLOCKED" "$OUTPUT_DIR/arthas-output.txt" | \
    awk '{print $1}' | sort | uniq -c | sort -rn

# 检查死锁
echo ""
echo "2. 死锁检查:"
if grep -q "Found.*deadlock" "$OUTPUT_DIR/arthas-output.txt"; then
    echo "  ❌ 发现死锁!"
    grep -A 20 "Found.*deadlock" "$OUTPUT_DIR/arthas-output.txt"
else
    echo "  ✅ 未发现死锁"
fi

# 提取最忙线程
echo ""
echo "3. 最忙线程 (Top 5):"
grep -A 3 "top 5 busy thread" "$OUTPUT_DIR/arthas-output.txt" | head -20

# 统计线程类型
echo ""
echo "4. 线程类型分布:"
grep "Thread Name:" "$OUTPUT_DIR/arthas-output.txt" | \
    sed 's/.*Thread Name: //' | \
    sed 's/-[0-9]*$//' | \
    sort | uniq -c | sort -rn | head -10

# GC 信息
echo ""
echo "5. GC 统计:"
grep -A 5 "GC" "$OUTPUT_DIR/arthas-output.txt" | grep -v "^--$"

# 内存使用
echo ""
echo "6. 内存使用:"
grep -A 10 "heap" "$OUTPUT_DIR/arthas-output.txt" | head -15

echo ""
echo "=" | awk '{for(i=1;i<=80;i++)printf "="; printf "\n"}'
echo "【详细报告】"
echo "=" | awk '{for(i=1;i<=80;i++)printf "="; printf "\n"}'
echo ""
echo "完整输出: $OUTPUT_DIR/arthas-output.txt"
echo ""

# 生成建议
cat > "$OUTPUT_DIR/recommendations.md" <<EOF
# 线程分析报告

**生成时间**: $(date)
**进程PID**: $PID

## 诊断结果

### 线程状态分布
\`\`\`
$(grep -E "RUNNABLE|WAITING|TIMED_WAITING|BLOCKED" "$OUTPUT_DIR/arthas-output.txt" | \
    awk '{print $1}' | sort | uniq -c | sort -rn)
\`\`\`

### 建议

#### 如果 WAITING/TIMED_WAITING 线程过多 (>100):

1. **检查 RabbitMQ 配置**
   \`\`\`yaml
   spring:
     rabbitmq:
       listener:
         simple:
           concurrency: 2
           max-concurrency: 10
   \`\`\`

2. **优化 Tomcat 线程池**
   \`\`\`yaml
   server:
     tomcat:
       threads:
         max: 200
         min-spare: 10
   \`\`\`

3. **调整 Redisson 线程数**
   \`\`\`java
   config.setNettyThreads(16);
   config.setThreads(8);
   \`\`\`

#### 如果 RUNNABLE 线程过多 (>50):

1. 检查是否有CPU密集型任务
2. 使用 \`thread -n 10\` 查看最忙线程
3. 考虑优化算法或添加缓存

#### 如果发现 BLOCKED 线程:

1. 使用 \`thread -b\` 检查死锁
2. 分析锁竞争情况
3. 优化同步代码块

## Arthas 常用命令

\`\`\`bash
# 实时监控线程
dashboard

# 查看指定线程堆栈
thread <thread-id>

# 监控方法调用
monitor -c 5 com.example.Service method

# 追踪方法调用
trace com.example.Service method

# 反编译类
jad com.example.Service

# 查看方法参数和返回值
watch com.example.Service method "{params,returnObj}" -x 2
\`\`\`

## 参考资料

- Arthas 官方文档: https://arthas.aliyun.com/
- JVM 线程状态: https://docs.oracle.com/javase/8/docs/api/java/lang/Thread.State.html
EOF

echo "📝 诊断建议: $OUTPUT_DIR/recommendations.md"
echo ""
echo "💡 下一步:"
echo "   1. 查看详细报告: cat $OUTPUT_DIR/recommendations.md"
echo "   2. 如需进一步分析,可手动连接: as.sh $PID"
echo ""
