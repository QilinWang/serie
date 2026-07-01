[← 返回 README](../README.zh-CN.md)

## 命令行快速上手

安装后会得到一个命令：

```bash
serie
```

命令结构是：

```bash
serie <subcommand> <source> --option value
```

例如：

```bash
serie generate lorenz --shock param --sigma 0.1
```

读作：

- `serie`：运行 SERIE CLI；
- `generate`：生成一个 SERIE dataset cell；
- `lorenz`：使用 Lorenz source；
- `--shock param`：使用内置参数冲击干预；
- `--sigma 0.1`：观测噪声剂量；
- `generate`：始终输出 JSON，方便人和脚本读取。

第一次安装后，先跑这三条：

```bash
serie list-data-ids
serie demo lorenz --out-html tmp/lorenz_demo.html
serie generate lorenz --sigma 0.1
```

它们分别做三件事：

1. 看看有哪些内置 source；
2. 生成一个离线交互式 Lorenz attractor HTML；
3. 生成一个 dataset cell，并返回 train/validation/test 切分摘要。

如果你想做完整安装检查，再跑：

```bash
serie list-cli-commands --format json
serie show-data-dir --format json
serie generate ou --sigma 0.1
serie generate lorenz --shock param --sigma 0.1 --out tmp/lorenz_cell --array-format all --plot both
serie walkthrough --out tmp/walkthrough --array-format all --overwrite --format json
```

## 常用 CLI

### 查看命令列表

```bash
serie list-cli-commands
serie list-cli-commands --format json
```

### 查看内置数据源

```bash
serie list-data-ids --format json
```

### 生成一个 SERIE cell

```bash
serie generate lorenz --sigma 0.25
serie generate lorenz --shock param --sigma 0.1 --out tmp/lorenz_cell --array-format csv --plot both
serie generate ou --sigma 1.0 --sigma-scale train_std
```

`generate` 是主要 CLI：它会生成 source，按需要加入干预，注入观测噪声，切分 train/validation/test，并且可以选择写出数组和图。输出 JSON 会包含 DGP bank 摘要和 cell 切分摘要。

- `--shock none|param|state|switch`：选择干预类型。比如 `lorenz --shock param` 会解析成 `lorenz_param_shock`。
- `--sigma`：观测噪声剂量。
- `--sigma-scale unit`：直接使用 `sigma`。
- `--sigma-scale train_std`：按训练段每个通道的标准差缩放噪声，适合不同通道有不同物理单位的真实记录。这个模式目前用于 JSON 切分摘要；如果要同时写数组或图，请使用 `--sigma-scale unit`。
- `--out`：写文件的根目录。不传 `--out` 时只打印 JSON 摘要；传入后文件会写到稳定的 `<source>_<cell-hash>` 子目录。
- `--array-format none|npy|csv|ipc|parquet|table|safetensors|all`：写出数组和 `manifest.json`。`safetensors` 会写一个 grouped `dgp_bank.safetensors`，里面包含 `clean`、`noise` 和可选的 `noisy_sigma_*` 张量；`all` 也包含这个 grouped 文件。
- `--plot none|timeseries|attractor|both`：写出静态图。

### 生成交互式 HTML demo

```bash
serie demo lorenz --out-html tmp/lorenz_demo.html
serie demo rossler --out-html tmp/rossler_demo.html --n-draws 30 --overwrite
serie demo chua --out-html tmp/chua_demo.html --n-points 4000 --n-draws 30 --overwrite
serie demo henon --out-dir tmp/henon_demo_pack --overwrite
```

生成的是一个单文件 HTML。浏览器里可以拖拽 3D attractor，滑动 `sigma` 实时看到：

```text
clean + sigma * noise_draw
```

这里的 `sigma` 是 unit-noise scale 下的自然噪声倍率。`Noise realization` 下拉框选择的是第几个冻结噪声 realization；同一个 realization 会在不同 `sigma` 下重复使用。

这些 HTML 很适合放到 GitHub Release 或静态网站作为展示材料，但不应该打进 PyPI wheel。PyPI 包里只带一个小的 viewer template，真正的数据由 `serie demo` 在本地生成后内联进 HTML。

如果你想一次得到一个检查用文件夹，用 `--out-dir`。它会写出
`index.html`、`arrays/` 里的 IPC 数组，以及 `static/` 里的 PNG 图。只有在你
额外传 `--data-out path/to/demo.json` 时，才会单独写出 browser JSON payload。
普通 `--out-html` 仍然是最适合分享的单文件 HTML。

