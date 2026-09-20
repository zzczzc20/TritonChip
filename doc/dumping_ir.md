# 看编译过程：IR 与汇编的 dump

和主流 Triton 一样：编译的每个阶段都可以写到磁盘上，也可以直接从
`CompileResult.asm` 里取。名字沿用 Triton 的习惯，因为已经调过 Triton
kernel 的人会先去找那几个开关。

## 开关

| | TNCA | Triton 里的对应物 |
|---|---|---|
| 打开 dump | `TNCA_KERNEL_DUMP=1` | `TRITON_KERNEL_DUMP=1` |
| 写到哪里 | `TNCA_DUMP_DIR=<path>`（默认 `~/.tnca/dump`） | `TRITON_DUMP_DIR` |
| 绕过缓存 | `TNCA_ALWAYS_COMPILE=1` | `TRITON_ALWAYS_COMPILE=1` |
| 单次调用 | `compile_aot(..., options={'dump_dir': '...'})` | — |
| 命令行 | `tnca-aot compile --dump-dir DIR` | — |

## 阶段

按产生的先后顺序：

| 文件 | 内容 |
|---|---|
| `<kernel>.ttir` | Triton 前端产出的 TTIR。**只有 Triton 源才有**：`tnca-ir` 程序从来没有 TTIR，给它写一个空文件会让人以为它有 |
| `<kernel>.tnir` | 导入后的 TNCA IR，寄存器分配之前 |
| `<kernel>.s` | 汇编。扩展名就是汇编文件该用的那个 |
| `<kernel>.hex` | 机器码镜像，一行一个 64 位字——R1 的工具和设备 bench 直接读这个格式 |
| `<kernel>.json` | 上面几个文件分别是什么，以及是什么编出来的 |

每个 kernel 一个目录（`<dump-dir>/<kernel>/`），所以两个 kernel 不会互相
覆盖，而同一个 kernel 重编会覆盖自己。

## 命令行

```
$ tnca-aot compile kern.py:axpy -S \
      --signature x_ptr=*fp32 ... --constexpr BLOCK=16 -o axpy.tnca
    s.rdspecial  s1, arg_base
    s.ld.d       s2, [s1, #0]      // x_ptr: device_ptr64 at arg_base+0
    ...
```

`-S` 和 `--emit` 跟着大家已经在用的编译器走：`gcc -S` 出汇编，
`clang -emit-llvm` 出 IR。

* `-S` — 等价于 `--emit asm`
* `--emit ttir|tnir|asm|hex` — 把某一个阶段打到 stdout，**不写文件**：
  要求「看」和要求「存」是两件事
* `--dump-dir DIR` — 全部阶段写进 `DIR/<kernel>/`

## Python

```python
res = aot.compile_aot(kernel, signature=..., constexprs=...)
print(res.asm['tnir'])        # 和 Triton 的 kernel.asm['ttgir'] 同一个意思
print(res.asm.keys())         # ['ttir', 'tnir', 'asm', 'hex']
res.dumped                    # 阶段 -> 已写出的文件路径；没开 dump 时是空的
```

`res.asm` **总是**有内容，不管有没有往磁盘写——这些文本在每个阶段结束的
那一刻本来就在内存里，留一个引用不花什么。

## 两条边界，都值得写下来

**dump 不改变 artifact。** `dump_dir` 和 `debug` 不进 cache key，也不进
manifest 的 `identity.options`。否则打开 dump 会给你一个和你想看的那个
不一样的 kernel，而那正是看它的理由。Triton 把 `TRITON_KERNEL_DUMP` 排除在
cache key 之外，是同一个道理。

顺带堵掉的一个问题：`dump_dir` 曾经会被写进 manifest，也就是把某个人的
home 目录路径打包进一个要分发的 artifact 里。

**要 dump 就一定会真编一次。** 缓存命中的意思是「别跑编译器」，dump 的
意思是「我要看编译器跑」。两个一起要的时候，编译会发生——否则一个 session
里第一次运行写出文件、后面每一次都悄悄不写，这是最让人困惑的行为。
`TNCA_ALWAYS_COMPILE=1` 是 Triton 里同名开关的对应物。

## 测试

`make r2-dump-tests`（DUMP-01…06）。这些**不是合同用例**：
`doc/phase2_tasks_and_acceptance.md` 没有要求 dump 功能，所以它们不计入
R2 的 84 条，文件开头也写明了这一点。
