# LLM 连接池使用指南

## 🎯 为什么需要连接池？

### 优化前的问题
```python
# ❌ 每次调用都创建新实例
def some_node(state):
    llm = get_llm()  # 创建实例1
    result = llm.invoke(...)

def another_node(state):
    llm = get_llm()  # 创建实例2（浪费！）
    result = llm.invoke(...)
```

### 优化后的优势
```python
# ✅ 复用实例，减少开销
全局池：[实例1, 实例2, 实例3]  (预创建，随时可用)

Node A → 借用实例1 → 归还
Node B → 复用实例1 → 归还  (无需创建！)
Node C → 借用实例2 → 归还
```

---

## 📊 连接池核心特性

### 1. **资源复用**
- 预创建 3 个实例（可配置）
- 自动借用/归还
- 零等待时间（池内有空闲实例时）

### 2. **线程安全**
- 使用 `threading.Lock` 保护共享状态
- 使用 `Queue` 管理实例队列
- 支持多线程并发访问

### 3. **智能扩展**
```
情况1: 池内有空闲实例 → 立即返回
情况2: 池为空 + 未达上限 → 创建新实例
情况3: 池为空 + 已达上限 → 阻塞等待归还
```

### 4. **自动管理**
使用上下文管理器，无需手动归还：
```python
with acquire_llm() as llm:
    result = llm.invoke(...)
# 自动归还到池中
```

---

## 🚀 使用方式

### 方式一：推荐使用（上下文管理器）

```python
from contextlib import contextmanager

def plan_research_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """使用 with 语句自动管理"""
    with acquire_llm() as llm:
        response = llm.invoke([HumanMessage(content=prompt)])
        # ... 处理响应
    # llm 实例自动归还到池中
    return state
```

**优点**:
- ✅ 自动归还，不会泄漏
- ✅ 异常安全（即使出错也会归还）
- ✅ 代码简洁

### 方式二：手动管理（兼容旧代码）

```python
def some_node(state):
    llm = get_llm()  # 从池中获取
    try:
        result = llm.invoke(...)
    finally:
        # ⚠️ 需要手动归还（容易忘记！）
        _llm_pool.release(cache_key, llm)
    return result
```

**缺点**:
- ❌ 需要手动归还
- ❌ 需要知道 cache_key
- ❌ 容易忘记导致资源泄漏

---

## 📈 性能对比

### 场景：5 个节点顺序执行

#### 优化前（简单缓存）
```
创建实例1 (100ms) → 使用 → 缓存
复用实例1 (0ms)   → 使用
复用实例1 (0ms)   → 使用
复用实例1 (0ms)   → 使用
复用实例1 (0ms)   → 使用
---
总创建时间: 100ms
总实例数: 1 个
```

#### 优化后（连接池）
```
初始化池 (300ms 一次性) → [实例1, 实例2, 实例3]

Node A → 借用实例1 (0ms) → 归还
Node B → 借用实例1 (0ms) → 归还  (复用！)
Node C → 借用实例1 (0ms) → 归还
Node D → 借用实例1 (0ms) → 归还
Node E → 借用实例1 (0ms) → 归还
---
总创建时间: 300ms (一次性)
总实例数: 3 个（并发时可同时使用）
```

### 场景：3 个节点并行执行

#### 简单缓存（只有1个实例）
```
Node A → 使用实例1 --------→
Node B → 等待... 等待... 等待... → 使用实例1 ------→
Node C → 等待... 等待... 等待... 等待... 等待... → 使用实例1 →
---
总耗时: T1 + T2 + T3 (串行)
```

#### 连接池（有3个实例）
```
Node A → 使用实例1 ------→
Node B → 使用实例2 ------→  (并行！)
Node C → 使用实例3 ------→  (并行！)
---
总耗时: max(T1, T2, T3) (并行)
```

**加速比**: 约 **3倍** （在并行场景下）

---

## 🔍 实时监控

```python
# 查看连接池统计
stats = _llm_pool.get_stats()
print(stats)

# 输出示例:
{
    ('openai', 'gpt-4o-mini', 0.2): {
        'total': 3,       # 总共创建的实例数
        'available': 2,   # 池内空闲实例数
        'in_use': 1       # 正在使用的实例数
    }
}
```

---

## ⚙️ 配置调优

### 调整池大小

```python
# 修改全局池大小（在初始化时）
_llm_pool = LLMPool(pool_size=5)  # 增加到 5 个实例

# 适用场景:
# - pool_size=1: 单线程场景（最省资源）
# - pool_size=3: 默认推荐（平衡性能与资源）
# - pool_size=10: 高并发场景（多线程/异步）
```

### 根据负载调整

| 场景 | 推荐池大小 | 原因 |
|------|-----------|------|
| 单任务顺序执行 | 1-2 | 无并发需求 |
| 中等并发（3-5个任务） | 3-5 | 平衡复用与并发 |
| 高并发（10+任务） | 10+ | 最大化并行能力 |
| Web 服务 | CPU核心数 × 2 | 标准线程池配置 |

---

## 🛡️ 最佳实践

### ✅ DO（推荐）

```python
# 1. 使用 with 语句
with acquire_llm(temperature=0.5) as llm:
    result = llm.invoke(messages)

# 2. 异常处理
with acquire_llm() as llm:
    try:
        result = llm.invoke(messages)
    except Exception as e:
        print(f"LLM调用失败: {e}")
        # llm 仍会自动归还
```

### ❌ DON'T（避免）

```python
# ❌ 忘记归还
llm = get_llm()
result = llm.invoke(messages)
# 忘记归还 → 资源泄漏！

# ❌ 重复获取
llm1 = get_llm()
llm2 = get_llm()  # 又创建一个？
# 应该复用 llm1

# ❌ 长时间占用
with acquire_llm() as llm:
    time.sleep(3600)  # 占用1小时 → 阻塞其他调用！
    result = llm.invoke(...)
```

---

## 🔧 故障排查

### 问题1: 卡住不动（阻塞）

**原因**: 所有实例都在使用中，且未归还

**解决**:
```python
# 检查是否有未归还的实例
stats = _llm_pool.get_stats()
print(stats)  # 如果 in_use 一直不减少 → 有泄漏

# 增加池大小
_llm_pool = LLMPool(pool_size=10)
```

### 问题2: 创建过多实例

**原因**: 并发调用超过池大小

**解决**:
```python
# 方案1: 增加池大小
_llm_pool = LLMPool(pool_size=20)

# 方案2: 限制并发数
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=3) as executor:
    # 最多3个线程并发
    results = executor.map(process_task, tasks)
```

### 问题3: 不同 temperature 需求

**场景**: 某些节点需要高创造性（temp=0.8），某些需要稳定（temp=0）

**解决**:
```python
# 连接池会为每个配置创建独立的池
with acquire_llm(temperature=0.8) as llm:  # 创造性任务
    creative_result = llm.invoke(...)

with acquire_llm(temperature=0.0) as llm:  # 稳定任务
    stable_result = llm.invoke(...)

# 统计会显示两个独立的池:
# ('openai', 'gpt-4o-mini', 0.8): {total: 3, ...}
# ('openai', 'gpt-4o-mini', 0.0): {total: 3, ...}
```

---

## 📚 总结

| 特性 | 简单缓存 | 连接池 |
|------|---------|--------|
| 实例复用 | ✅ | ✅ |
| 并发支持 | ❌ (1个实例) | ✅ (N个实例) |
| 线程安全 | ❌ | ✅ |
| 资源限制 | ❌ | ✅ |
| 自动归还 | N/A | ✅ |
| 性能监控 | ❌ | ✅ |

**适用场景**:
- ✅ **顺序执行** → 简单缓存即可
- ✅ **并发执行** → 必须使用连接池
- ✅ **生产环境** → 强烈推荐连接池
