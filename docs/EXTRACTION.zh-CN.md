# 中文阅读与功能提取指南

[English](EXTRACTION.md) | 中文

## 1. 这些文件适合怎样使用

这份仓库适合师兄和他的代码助手阅读历史实现、确认约束，再把选定功能接入另一份代码。它不是现行训练框架，也不是可以整份覆盖目标项目的补丁包。

现有材料的分工是：[首页](../README.md)讲范围与开关，[架构说明](ARCHITECTURE.md)讲谁负责什么以及执行顺序，[审阅指引](REVIEW.md)给源码入口，[检查记录](../VERIFICATION.md)说明实际验证到了哪里。本指南补上“选一个功能后，哪些可以提取、哪些必须重新接入、怎样证明没有改坏”。

建议先读本页，再按选中的功能阅读源码与测试，不必从头读完整训练框架。

## 2. 按功能选阅读范围

| 想提取的功能 | 先读哪里 | 提取边界 |
|---|---|---|
| 记录失败原因与执行阶段 | [记录工具](../verl/experimental/one_step_off_policy/trace_utils.py)、[错误说明测试](../tests/experimental/one_step_off_policy/test_trace_span_error_message_on_cpu.py) | 可先独立提取记录模块；目标代码自己选择包住哪些操作。下节给出一个真实提交的例子。 |
| 计算哪些生成引擎可以保留 | [区域与引擎比较](../verl/experimental/one_step_off_policy/awake_topology.py)、[算法测试](../tests/experimental/one_step_off_policy/test_awake_topology_on_cpu.py) | 模块只依赖 Python 标准库；目标代码必须提供一致的显卡分组及顺序。逻辑编号改变不等于物理引擎改变。 |
| 持有睡眠中的生成引擎 | [睡眠池](../verl/experimental/one_step_off_policy/elastic_engine_pool.py)、[池测试](../tests/experimental/one_step_off_policy/test_elastic_engine_pool_on_cpu.py)、[架构说明](ARCHITECTURE.md) | 模块可独立加载，但实际运行依赖目标代码的远程工作进程、异步完成通知及副本管理。不能只复制一个文件就宣称显卡睡眠流程可用。 |
| 不建立纯生成进程不用的训练模型壳 | [跳过条件](../verl/experimental/one_step_off_policy/shell_skip.py)、[条件测试](../tests/experimental/one_step_off_policy/test_skip_rollout_fsdp_shell_on_cpu.py)、[工作进程接入](../verl/experimental/one_step_off_policy/fsdp_workers.py)、[配置传递](../verl/experimental/one_step_off_policy/config/one_step_off_ppo_trainer_dynamic_resize.yaml) | 判断条件可单独提取；必须重新核实目标进程是否参与训练、是否还有模型壳消费者，以及开关是否传到了工作进程。训练进程始终不能跳过。 |
| 接入存活引擎、睡眠池或生成重叠流程 | [训练调度](../verl/experimental/one_step_off_policy/ray_trainer.py)、[七个审阅单元](REVIEW.md)、[架构说明](ARCHITECTURE.md) | 这是手工接入边界，不是独立模块。按目标框架已有的工作进程、权重同步和副本所有者重新实现；不要整份覆盖训练调度文件。 |

区域物理顺序是：**固定训练区 → 可转换区 → 固定生成区**。睡眠中的进程不能继续留在活跃的权重同步成员列表里；唤醒完成也不能代替权重同步。

## 3. 一个真实提交：失败时保留错误说明

**原始开发提交：`4b39d5861`，2026-08-22。** 它是八月源快照的末尾提交，不是本公开仓库的新提交。公开材料没有原始开发历史，不能在这里对这个编号执行检出或直接拣选提交。以下链接指向公开的最终源码和已净化测试，不是私有原始差异或实验附件。

### 改了什么，为什么值得先看它

以前阶段记录只保存错误类型：知道“出了什么类别的问题”，但丢了错误的具体说明。这次改动在已有的阶段记录工具上补充错误说明，并为边界情况加入测试。它提高诊断信息的完整性，不是加速或训练正确性的证明。

| 文件 | 这次改动的职责 |
|---|---|
| [记录工具](../verl/experimental/one_step_off_policy/trace_utils.py) | 安全地取得错误文字；过长文字保留前 2000 个字符并加截断标记；在记录中增加 `error_message`；中断也按失败记录；原异常仍向调用方抛出。 |
| [错误说明测试](../tests/experimental/one_step_off_policy/test_trace_span_error_message_on_cpu.py) | 检查错误说明保留、正常执行时字段为空、长说明截断、无法转成文字的异常不会盖住原异常、中断仍会抛出。公开测试中的示例数值是合成数据。 |

源码中的关键名字：`ResizeTraceConfig` 是记录开关与输出位置；`resize_trace_span` 包住一段操作并记录结果；`error_type` 是错误类型，`error_message` 是具体错误说明。

不要把整个记录模块都算作这一次提交的新功能：阶段计时、写文件和可选的显卡时间线标记已有实现。这次是在已有记录路径上补齐失败信息。

### 提取时要一起保留什么

1. 如果目标项目没有相同记录工具，先提取整个 `trace_utils.py`，保留适用的许可说明；不要导入整套训练包。
2. 如果目标项目已有记录工具，只迁移“取得说明、截断、记录字段、原异常透传”的行为，使用它已有的记录入口，避免再建一套并行工具。
3. 由目标代码把阶段记录接到自己的扩缩容或初始化操作。公开源码中的一个真实调用点是[训练状态恢复操作](../verl/experimental/one_step_off_policy/fsdp_workers.py#L2723-L2730)：它把实际操作包在阶段记录中，并传入角色信息。最小调用形状见下例；这里不复制旧训练调度文件。
4. 直接创建记录配置时不需要额外配置库。调用 `build_resize_trace_config` 才需要 `omegaconf`；开启 `VERL_RESIZE_NVTX` 才会尝试加载 `torch` 做显卡时间线标记。本例显式关闭这一可选路径。
5. 记录写入失败不应盖住业务异常。它是尽力记录，不是保证落盘的审计系统。
6. 新生成的记录会带上当前机器名、进程编号和错误原文。错误原文可能包含路径或秘密；示例只输出三个安全字段，临时记录自动删除。真实记录公开前必须另行检查。

### 可直接运行的小例子

在本仓库根目录执行下面的 Python 代码；它按文件加载实际模块，不启动训练，也不需要显卡。移植后，应换成目标项目的正常模块导入方式。

```python
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["VERL_RESIZE_NVTX"] = "0"
source = Path("verl/experimental/one_step_off_policy/trace_utils.py")
spec = importlib.util.spec_from_file_location("trace_example", source)
trace = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trace
spec.loader.exec_module(trace)

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "trace.jsonl"
    config = trace.ResizeTraceConfig(enabled=True, path=str(path))
    original = RuntimeError("example resize failed")
    try:
        with trace.resize_trace_span(config, "example_resize"):
            raise original
    except RuntimeError as caught:
        assert caught is original
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["status"] == "error"
    assert record["error_type"] == "RuntimeError"
    assert record["error_message"] == "example resize failed"
    print({key: record[key] for key in ("status", "error_type", "error_message")})
```

输出应为：

```text
{'status': 'error', 'error_type': 'RuntimeError', 'error_message': 'example resize failed'}
```

这说明具体错误说明被留下，而且原异常没有被替换或吞掉。它不证明训练、显卡内存回收或多机运行。

运行对应的五个现有行为检查：

```bash
python3 -m pytest -q tests/experimental/one_step_off_policy/test_trace_span_error_message_on_cpu.py
```

`pytest` 是运行检查所需的工具；前面的独立例子不依赖它。把功能接入目标代码后，还需在那里重做这些边界检查，并实际触发一次目标操作的失败路径。

如果只提取记录模块与这个测试文件，请保留它们当前的相对目录布局，或修改测试中的源码定位方式；不需要搬走整套训练依赖。

## 4. 可交给师兄助手的任务说明

下面是一段可直接转交的阅读任务，不是已经完成移植的声明。

> 请先读本仓库的中文功能提取指南和对应英文页，选择一个功能，再读该功能的源码、现有行为检查以及目标项目的实际调用点。先只读，不修改。
>
> 请给出：① 功能解决什么问题及不变的行为；② 最小文件范围、依赖和开关；③ 目标项目已有的负责模块及需要重接的调用点；④ 能直接保留与必须重新实现的部分；⑤ 失败、取消和执行顺序的风险；⑥ 目标代码中的验证方法与尚未验证的边界。
>
> 可以从原始提交 `4b39d5861` 对应的“失败时保留错误说明”开始。公开链接展示最终实现，不是可直接拣选的原始提交。不要请求或复制原始私人记录、服务器配置或实验附件；不要整份覆盖目标训练调度，也不要把纯算法检查说成训练复现。实现前先给出目标调用点的接入方案。

## 5. 公开与验证边界

- [工作时间线](WORK_HISTORY.md)区分实际开发日期和公开日期；不伪造开发历史。
- 原始提交身份、完整差异与实验资料保持私有。本页仅公开一个经核实的功能例子及最终源码入口。
- 本次例子使用合成错误文字和临时目录，不发布机器记录或实测数字。
- [检查记录](../VERIFICATION.md)是实际运行结果的入口。八月代码不是当前团队主线，也没有验证迁移到任意新框架后仍然正确。
