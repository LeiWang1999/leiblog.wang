---
title: AMOS Analysis
categories:
  - Technical
tags:
  - Digilal Design
  - EEEE
date: 2022-08-25 16:47:30
---

<!-- more -->

```python
measure_opt = at.MeasureOptions(target=target, timeout=100, number=200, min_repeat_ms=500)
```

```python
class MeasureOptions(object):
    def __init__(
        self,
        target="llvm",
        build_func="default",
        target_host="llvm",
        timeout=10,
        verbose=1,
        number=100,
        repeat=1,
        min_repeat_ms=150,
        cooldown_interval=0,
        enable_cpu_cache_flush=1,
        dev_id=0,
        use_rpc=False,
        key=None,
        host=None,
        port=None,
        priority=1,
    ):
        self.target = target
        self.build_func = build_func
        self.target_host = target_host
        self.timeout = timeout
        self.verbose = verbose
        self.number = number
        self.repeat = repeat
        self.min_repeat_ms = min_repeat_ms
        self.cooldown_interval = cooldown_interval
        self.enable_cpu_cache_flush = enable_cpu_cache_flush
        self.dev_id = dev_id
        self.use_rpc = use_rpc
        self.key = key
        self.host = host
        self.port = port
        self.priority = priority
```

auto_tensorize_compute:



q: 

1. target_dag 拿来干什么用的？
