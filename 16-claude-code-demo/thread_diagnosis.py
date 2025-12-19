"""
线程等待问题排查工具

使用方法:
1. 在 Arthas 中执行: thread --all > threads.txt
2. 运行此脚本分析: python thread_diagnosis.py
"""

import re
from collections import defaultdict, Counter
from typing import Dict, List, Set


class ThreadAnalyzer:
    """线程分析器"""

    def __init__(self):
        self.threads = []
        self.waiting_threads = []
        self.timed_waiting_threads = []
        self.runnable_threads = []
        self.blocked_threads = []

    def analyze_arthas_output(self, content: str):
        """分析 Arthas thread 命令输出"""
        print("=" * 80)
        print("线程等待问题诊断报告")
        print("=" * 80)

        # 1. 统计线程状态
        print("\n【1. 线程状态统计】")
        state_counts = {
            'WAITING': content.count('WAITING'),
            'TIMED_WAITING': content.count('TIMED_WAITING'),
            'RUNNABLE': content.count('RUNNABLE'),
            'BLOCKED': content.count('BLOCKED'),
        }

        for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {state:20s}: {count:4d} 个线程")

        # 2. 分析等待原因
        print("\n【2. 等待原因分析】")

        # RabbitMQ 线程
        rabbit_waiting = content.count('org.springframework.amqp.rabbit') + \
                        content.count('RabbitListener')
        if rabbit_waiting > 0:
            print(f"  ⚠️  RabbitMQ 消费者线程等待: {rabbit_waiting} 个")
            print(f"      原因: 队列无消息,线程空闲等待")
            print(f"      建议: 正常现象,可考虑减少 consumer 并发数")

        # HTTP 线程池
        http_waiting = content.count('http-nio') + content.count('HTTP')
        if http_waiting > 0:
            print(f"  ⚠️  HTTP 线程池等待: {http_waiting} 个")
            print(f"      原因: 无请求时线程空闲")
            print(f"      建议: 检查 server.tomcat.threads.max 配置")

        # Nacos 线程
        nacos_waiting = content.count('com.alibaba.nacos')
        if nacos_waiting > 0:
            print(f"  ⚠️  Nacos 客户端线程: {nacos_waiting} 个")
            print(f"      原因: 配置中心/注册中心心跳线程")
            print(f"      建议: 正常现象")

        # Redisson 线程
        redisson_waiting = content.count('redisson-netty')
        if redisson_waiting > 0:
            print(f"  ⚠️  Redisson Netty 线程: {redisson_waiting} 个")
            print(f"      原因: Redis 客户端 NIO 线程")
            print(f"      建议: 检查 Redisson 连接池配置")

        # Async 线程
        async_waiting = content.count('AsyncTraceDispatcher') + \
                       content.count('async')
        if async_waiting > 0:
            print(f"  ⚠️  异步任务线程: {async_waiting} 个")
            print(f"      原因: 异步任务完成后等待新任务")
            print(f"      建议: 检查 @Async 线程池配置")

        # 3. 识别潜在问题
        print("\n【3. 潜在问题识别】")

        issues = []

        # 检查是否有死锁
        if 'BLOCKED' in content:
            issues.append("❌ 发现 BLOCKED 线程 - 可能存在死锁或锁竞争")

        # 检查线程数是否过多
        total_threads = sum(state_counts.values())
        if total_threads > 200:
            issues.append(f"❌ 线程总数过多 ({total_threads}) - 可能导致性能问题")

        # 检查是否有过多 RUNNABLE 线程
        if state_counts.get('RUNNABLE', 0) > 50:
            issues.append(f"❌ RUNNABLE 线程过多 ({state_counts['RUNNABLE']}) - CPU 可能过载")

        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  ✅ 未发现明显问题")

        return state_counts


class ThreadDumpAnalyzer:
    """线程堆栈分析器"""

    @staticmethod
    def parse_thread_dump(dump_text: str) -> List[Dict]:
        """解析线程堆栈信息"""
        threads = []
        current_thread = None

        for line in dump_text.split('\n'):
            # 匹配线程名称行
            if line.startswith('"'):
                if current_thread:
                    threads.append(current_thread)

                # 提取线程名称和状态
                match = re.search(r'"([^"]+)".*State:\s*(\w+)', line)
                if match:
                    current_thread = {
                        'name': match.group(1),
                        'state': match.group(2),
                        'stack': []
                    }
            elif current_thread and line.strip().startswith('at '):
                # 提取堆栈信息
                stack_line = line.strip()[3:]  # 去掉 "at "
                current_thread['stack'].append(stack_line)

        if current_thread:
            threads.append(current_thread)

        return threads

    @staticmethod
    def analyze_waiting_patterns(threads: List[Dict]):
        """分析等待模式"""
        print("\n【4. 等待模式分析】")

        wait_patterns = defaultdict(list)

        for thread in threads:
            if thread['state'] in ['WAITING', 'TIMED_WAITING']:
                # 找到等待的关键方法
                for stack_line in thread['stack'][:5]:  # 只看前5行
                    if any(keyword in stack_line for keyword in
                          ['wait', 'park', 'await', 'poll', 'take', 'sleep']):
                        wait_patterns[stack_line].append(thread['name'])
                        break

        # 按频率排序
        sorted_patterns = sorted(wait_patterns.items(),
                                key=lambda x: len(x[1]),
                                reverse=True)

        print(f"\n  发现 {len(sorted_patterns)} 种等待模式:")
        for i, (pattern, thread_names) in enumerate(sorted_patterns[:5], 1):
            print(f"\n  {i}. {pattern}")
            print(f"     线程数: {len(thread_names)}")
            print(f"     示例: {thread_names[0]}")


def generate_arthas_commands():
    """生成 Arthas 排查命令"""
    print("\n" + "=" * 80)
    print("【排查步骤】")
    print("=" * 80)

    commands = [
        ("1. 查看所有线程概览", "dashboard"),
        ("2. 导出完整线程堆栈", "thread --all > /tmp/threads.txt"),
        ("3. 查看最忙的3个线程", "thread -n 3"),
        ("4. 查看指定状态线程", "thread --state WAITING"),
        ("5. 查看死锁", "thread -b"),
        ("6. 监控线程池", "monitor -c 5 java.util.concurrent.ThreadPoolExecutor execute"),
    ]

    for desc, cmd in commands:
        print(f"\n{desc}")
        print(f"  arthas> {cmd}")


def generate_solutions():
    """生成解决方案"""
    print("\n" + "=" * 80)
    print("【解决方案建议】")
    print("=" * 80)

    solutions = {
        "1. RabbitMQ 消费者线程过多": """
  问题: 队列无消息,但保持大量空闲消费者
  解决:
    # application.yml
    spring:
      rabbitmq:
        listener:
          simple:
            concurrency: 2        # 最小消费者数
            max-concurrency: 10   # 最大消费者数
            prefetch: 1           # 每次拉取消息数
""",
        "2. HTTP 线程池配置不当": """
  问题: Tomcat 线程池过大,大量线程空闲
  解决:
    # application.yml
    server:
      tomcat:
        threads:
          max: 200        # 最大线程数
          min-spare: 10   # 最小空闲线程
        max-connections: 8192
        accept-count: 100
""",
        "3. Redisson 连接数过多": """
  问题: Netty 线程数过多
  解决:
    RedissonClient redisson = Redisson.create(
        Config.fromYAML(config)
            .setNettyThreads(16)  // 降低 Netty 线程数
            .setThreads(8)        // 降低业务线程数
    );
""",
        "4. 异步线程池配置": """
  问题: @Async 线程池过大
  解决:
    @Configuration
    @EnableAsync
    public class AsyncConfig implements AsyncConfigurer {
        @Override
        public Executor getAsyncExecutor() {
            ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
            executor.setCorePoolSize(10);      // 核心线程数
            executor.setMaxPoolSize(50);       // 最大线程数
            executor.setQueueCapacity(100);    // 队列大小
            executor.setThreadNamePrefix("async-");
            executor.setRejectedExecutionHandler(
                new ThreadPoolExecutor.CallerRunsPolicy()
            );
            executor.initialize();
            return executor;
        }
    }
""",
        "5. 定期监控": """
  使用 Prometheus + Grafana 监控线程指标:
    - JVM 线程总数
    - 各状态线程数量
    - 线程池队列长度
    - 线程池活跃线程数
"""
    }

    for title, solution in solutions.items():
        print(f"\n{title}")
        print(solution)


def main():
    """主函数"""
    print("\n🔍 线程等待问题排查工具\n")

    # 模拟分析
    sample_content = """
    TIMED_WAITING: org.springframework.amqp.rabbit.RabbitListener main
    TIMED_WAITING: http-nio-8081-exec-1
    TIMED_WAITING: redisson-netty-2-1
    WAITING: com.alibaba.nacos.client.Worker.fixed-1
    RUNNABLE: C2 CompilerThread0
    """

    analyzer = ThreadAnalyzer()
    analyzer.analyze_arthas_output(sample_content * 20)  # 模拟多个线程

    # 生成排查命令
    generate_arthas_commands()

    # 生成解决方案
    generate_solutions()

    print("\n" + "=" * 80)
    print("【快速诊断清单】")
    print("=" * 80)
    print("""
1. ✅ 检查线程总数是否超过 200
2. ✅ 确认 BLOCKED 线程数量 (应为 0)
3. ✅ 查看 RUNNABLE 线程占比 (正常 < 20%)
4. ✅ 检查线程池配置是否合理
5. ✅ 确认是否有死锁 (arthas: thread -b)
6. ✅ 监控 GC 情况 (arthas: dashboard)
7. ✅ 查看 CPU 使用率 (top -Hp <pid>)
8. ✅ 检查内存使用 (arthas: memory)
""")

    print("\n💡 提示: 大部分 WAITING/TIMED_WAITING 是正常的,")
    print("    只要线程总数合理、无死锁、CPU不高,就无需过度优化。\n")


if __name__ == '__main__':
    main()
