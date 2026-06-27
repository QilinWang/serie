# To Learn — SERIE 项目技能路线图

[English / 中文混合] 围绕 SERIE 项目可以学习和展示的技能，从最近的开始。

---

## Part 0: 当前深度待办与部署练习

这一段是给现在的项目节奏用的：先把 **SERIE 包本身**、**浏览器演示页**、
**发布流程** 三件事分清楚，再练部署。

### 当前状态板

| 事项 | 当前状态 | 下一步 |
|------|----------|--------|
| PyPI 包 | `serie` 已经可以构建 wheel/sdist，`just check` 会做本地完整检查 | 发布新版本前改版本号，跑 `just check`，再 `just publish-release` 或 GitHub Trusted Publishing |
| GitHub CI | `.github/workflows/ci.yml` 已经跑 ruff、pyright、pytest、wheel 检查 | push 后在 GitHub Actions 页看是否全绿 |
| GitHub Pages demo | `.github/workflows/pages-demo.yml` 已经生成一个离线 `serie demo shapes` HTML 并发布 | GitHub 仓库 Settings → Pages → Source 选 **GitHub Actions**，然后 push 或手动 Run workflow |
| Netlify demo | `demo/netlify.toml` 是 Vite/React 演示源码的部署入口 | 作为前端部署练习：让 Netlify 从 `demo/` 构建，或者只上传本地生成的静态 HTML |
| 腾讯云 demo | 最适合练“静态文件托管” | 跑 `just build-demo-static` 生成 `tmp/deploy_site/index.html`，上传到 COS 静态网站 bucket，object key 用 `index.html` |
| README / 教程 | README 面向用户，`to_learn.md` 面向你学习维护 | 不把 shell 魔法塞进 README；把概念解释留在这里 |

### 三种“demo”不要混在一起

SERIE 现在有三种相关但不同的东西：

```text
1. serie demo --out-html
   Python CLI 生成一个自包含 HTML 文件。最适合分享、GitHub Pages、腾讯 COS。
   本仓库的快捷命令是 just build-demo-static，默认写到 tmp/deploy_site/index.html。
   `serie demo shapes` 会把 Lorenz、Rössler、Chua 打进同一个 HTML，并在浏览器里给一个下拉菜单。

2. serie demo --out-dir
   Python CLI 生成一个检查文件夹：index.html + arrays/ + static plots。
   适合你检查数据，不适合作为默认发布网页。

3. serie/demo/
   React + Three.js + Vite 源码。它是前端练习场；需要 Bun/Node 工具链。
   PyPI wheel 不发布这个源码，只发布预构建的 demo_dist 模板。
```

### Astro project page 可以放在哪里

Astro 适合做 SERIE 的 **项目主页 / 文档型展示站**，但不应该替代现有的
`serie demo` 或 `serie/demo/`：

```text
serie demo --out-html     -> 生成一个可直接托管的交互式 demo HTML
serie/demo/               -> React + Three.js + Vite viewer 源码
Astro project page        -> 把 README、CLI 说明、研究动机、demo 链接组织成网站
```

心智模型：

```text
Astro 负责页面和内容结构：
  src/pages/index.astro       首页
  src/pages/demo.astro        demo 入口页
  src/pages/cli.astro         CLI 示例和 JSON summary 说明
  src/pages/deploy.astro      GitHub Pages / Netlify / 腾讯 COS 说明
  src/pages/research.astro    DGP bank、titration、exact benchmark 叙事

现有 SERIE demo 负责交互体验：
  just build-demo-static      生成 tmp/deploy_site/index.html
  serie demo shapes ...       生成可嵌入或链接的 attractor viewer
```

为什么不是只用 Vite/React？

```text
Vite/React 更适合做一个重交互应用。
Astro 更适合做多页面项目站：静态页面、Markdown/MDX 内容、少量交互岛。
SERIE 项目页主要是解释和导航，只有 demo 区域需要强交互，所以 Astro 更顺手。
```

最小目录建议：

```text
site/
  package.json
  astro.config.mjs
  src/
    pages/
      index.astro
      demo.astro
      cli.astro
      deploy.astro
      research.astro
  public/
    demo/
      index.html   # 可由 serie demo shapes --out-html 生成或复制进来
```

第一版不要做复杂集成。先把现有 demo 当静态资产放进 Astro：

```bash
just build-demo-static
mkdir -p site/public/demo
cp tmp/deploy_site/index.html site/public/demo/index.html
```

然后 `site/src/pages/demo.astro` 里先用链接或 iframe 指向 `/demo/index.html`。
等页面结构跑通后，再考虑把 `serie/demo/` 的 React viewer 抽成 Astro React island。

推荐 Justfile 名字：

```text
dev-project-page       # 本地启动 Astro dev server
build-project-page     # 构建 site/dist
check-project-page     # 构建并检查输出
```

部署心智模型：

```text
GitHub Pages / Netlify 发布 site/dist
腾讯 COS 仍然优先练 just build-demo-static 生成的单文件 index.html
```

官方文档入口：

```text
Astro project structure: https://docs.astro.build/en/basics/project-structure/
Astro React integration: https://docs.astro.build/en/guides/integrations-guide/react/
Astro on GitHub Pages:   https://docs.astro.build/en/guides/deploy/github/
Astro on Netlify:        https://docs.astro.build/en/guides/deploy/netlify/
```

### GitHub Pages 练习

这个已经是仓库里最标准的“自动部署”练习。你不需要选 GitHub 提供的
Jekyll / Static HTML 模板，因为仓库已经有自己的 workflow：

```text
.github/workflows/pages-demo.yml
```

操作顺序：

```bash
git add ...
git commit -m "..."
git push
```

然后在 GitHub 页面：

```text
Settings → Pages → Build and deployment → Source → GitHub Actions
Actions → Pages Demo → 等它跑完
```

这个 workflow 做的事是：

```text
checkout 代码
安装 uv
安装 Python 3.14
运行 serie demo shapes --out-html site/index.html
把 site/ 上传给 GitHub Pages
```

心智模型：**GitHub Pages 不是在跑 Python 服务。** 它只是托管最后生成出来的
静态 HTML 文件。

### Netlify 练习

Netlify 可以做两种不同练习：

#### A. 前端源码练习（推荐给你学 Vite/React）

把 Netlify 的 base directory 设成：

```text
demo
```

构建命令（`demo/netlify.toml` 已经写好）：

```bash
python -m pip install uv && bun run bake && bun run build
```

发布目录：

```text
dist
```

这条路会让 Netlify 构建 `serie/demo/` 这个 React/Vite 项目。它更适合学习：

```text
package.json
bun install
vite build
dist/
netlify.toml
```

注意：`python -m pip install uv` 是为了让 Netlify 的干净构建机能运行
SERIE 的 Python bake 步骤。`bun run bake` 会先用 SERIE Python API 生成浏览器需要的
`public/attractor_data.json`。如果只跑 `bun run build`，页面可能能打开，
但没有真实数据，只会显示“请先 bake JSON”的提示。

Netlify 官方文档里，`netlify.toml` 的 `[build]` 可以指定 `command` 和
`publish`；如果配置冲突，`netlify.toml` 会覆盖 UI 里的同名设置：
https://docs.netlify.com/build/configure-builds/file-based-configuration/

Netlify 也能自动识别 Bun 项目；如果 base directory 里是 Bun 项目
（例如有 `bun.lock` / `bun.lockb` 这样的 Bun lockfile），它会用 Bun 安装依赖：
https://docs.netlify.com/build/configure-builds/manage-dependencies/#bun

本机练习时要注意一个坑：`bunx netlify status` 这次在你的 WSL 里失败了，
因为 Netlify CLI 触发了 Node 专有的 compile-cache 路径，而 Bun 的兼容层没有完全覆盖。
同时你的 `/usr/local/bin/node` 实际上是指向 Bun 的 symlink，不是真正的 Node：

```text
/usr/local/bin/node -> /home/qilin/.bun/bin/bun
```

这解释了为什么 `npx netlify status` 找不到，`bunx netlify status` 又崩。
结论：

```text
前端项目构建：Bun 可以继续用。
Netlify CLI：最好安装真正的 Node/npm/npx 后再用。
不用 CLI 的练习：直接在 Netlify 网页 UI 上传 site/ 或连接 GitHub，也完全可以。
```

#### B. 静态 HTML 练习（推荐给 SERIE 用户）

本地或 CI 先生成：

```bash
just build-demo-static
```

然后 Netlify 只发布：

```text
tmp/deploy_site/
```

这条路更像 GitHub Pages：不教太多前端构建，但最接近“研究者想看一个
SERIE demo 文件”的真实需求。当前 `shapes` 组是三种原生 3D 形状系统：

```text
lorenz
rossler
chua
```

随机系统里目前没有原生 3D 源。OU、doublewell、SLDS、seasonal_ar 是 1D；
GARCH 是 2D，可以被补一个 z=0 通道放进 3D viewer，但它不是原生 3D attractor。

### 腾讯云 COS 静态网站练习

腾讯云 COS 最适合练“对象存储 + 静态网站”。它不需要运行 Python、
Node、Bun、Vite。你上传的是已经生成好的静态文件。

本地先做：

```bash
just build-demo-static
```

然后在腾讯云控制台：

```text
COS → Bucket 列表 → 创建 bucket
权限：public-read / private-write（公开读，私有写）
基础配置 → 静态网站 → 开启
索引文档：index.html
上传 tmp/deploy_site/index.html，object key 写成 index.html
访问静态网站地址或绑定自定义域名
```

这个命令和 `just build-dist-release` 没有关系：

```text
just build-demo-static  -> 生成给网页托管用的 tmp/deploy_site/index.html
just build-dist-release -> 生成给 PyPI 上传用的 dist/*.whl 和 dist/*.tar.gz
```

腾讯云官方 COS 静态网站文档：
https://www.tencentcloud.com/document/product/436/30958

中文文档：
https://cloud.tencent.com/document/product/436/32670

注意两个现实问题：

```text
1. 新 bucket 可能要求自定义域名才能正常网页预览；
2. 如果用中国大陆自定义域名，通常会涉及 ICP 备案。
```

所以第一次练习时，目标不是“完美正式上线”，而是：

```text
我能把 SERIE 生成的 index.html 作为静态网页放到一个云 bucket 上。
```

### 更深的项目待办

| 优先级 | 待办 | 原因 |
|--------|------|------|
| 高 | push 当前 SERIE 改动，让 CI / Pages 真跑一次 | 本地通过不等于云端通过 |
| 高 | 新版本发布前跑 `just check`，再确认 wheel 内容 | PyPI 文件不可覆盖，发布前要一次过 |
| 中 | Netlify 先走 `demo/` 源码路径 | 这是学习前端构建链最好的练习 |
| 中 | 腾讯 COS 先手动上传 `tmp/deploy_site/index.html` | 学静态托管，不把云密钥放进 repo |
| 中 | README 徽章确认都指向正确 workflow | 让项目入口显得正式，也能快速看到状态 |
| 后 | Netlify / 腾讯自动化上传 | 需要 token/secret；等手动流程跑通再自动化 |
| 后 | Cloudflare / EdgeOne 对比 | 真要中国访问速度，再评估 CDN/域名/备案 |

---

## Part 1: 你已有的 GitHub Actions 逐行解读

### 什么是 GitHub Actions？

GitHub Actions 是 GitHub 内置的自动化系统。你在仓库里放一个 `.github/workflows/*.yml`
文件，GitHub 就会在特定事件（push、PR、打 tag）发生时，自动启动一台云端虚拟机，
按你写的步骤执行命令。

核心概念只有四个：

```
事件 (on)  →  触发一个  工作流 (workflow)
工作流包含多个  作业 (job)
每个作业包含多个  步骤 (step)
每个步骤要么运行一条 shell 命令 (run)，要么调用一个别人写好的动作 (uses)
```

你可以把它想成：**"每次我 push 代码，GitHub 帮我开一台干净的 Linux 机器，
装好环境，跑测试，然后告诉我过没过。"**

---

### 本地 check、pre-commit、CI、publish 到底是什么关系？

这四个东西经常被混在一起，但它们其实是不同时间点的门禁：

| 名字 | 在哪里跑 | 什么时候跑 | 目的 |
|------|----------|------------|------|
| `just check` | 你的本机 | 你手动运行 | 最完整的本地检查：lint、type、coverage tests、wheel build、clean install smoke test |
| pre-commit hook | 你的本机 | `git commit` 前自动跑 | 提前挡住明显坏提交，通常只跑快检查 |
| GitHub Actions CI | GitHub 云端机器 | push / PR 后自动跑 | 在干净环境复现检查，证明"不是只在我电脑上能过" |
| `just publish-release` | 你的本机 | 你决定发布时手动跑 | 先跑完整检查，再把 wheel/sdist 上传 PyPI |

#### pre-commit hook 是什么？

Git 有一个机制叫 **hook**：在某些 Git 动作前后自动执行脚本。
`pre-commit` 就是 `git commit` 之前的 hook。

如果安装了 hook，你输入：

```bash
git commit -m "..."
```

Git 会先自动跑配置好的检查。检查失败，commit 就不会生成。

SERIE 现在的 `.pre-commit-config.yaml` 只配置了两个快检查：

```text
Ruff lint
Pyright
```

它不跑 pytest，因为测试更慢。完整检查仍然是：

```bash
just check
```

所以最简单的心智模型是：

```text
日常手动确认：just check
GitHub 自动确认：Actions / CI
可选本地自动挡板：pre-commit hook
发布上传：just publish-release
```

#### 为什么 Justfile 里可以有组合命令？

Justfile 有两种有价值的层次：

```text
大命令：你真正记住和运行的命令
小原语：被大命令复用的步骤
```

例如 `just check` 是大命令；它内部可以调用 `check-source`、`check-wheel`
这样的可见小步骤去 "build wheel → clean install → run installed CLI"。
这种组合是好的，因为：

```text
1. 你只记住一个入口；
2. 重复步骤有名字，维护时不复制粘贴；
3. `just --list` 能看到层级关系，学习时不会觉得有神秘步骤；
4. 但旧概念不能靠"隐藏"继续活着。
```

规则：如果一个 helper 没有复用价值，应该 inline/delete；如果有复用价值，
应该在 `just --list` 的注释里说清楚它属于哪个大命令、为什么存在。

---

### ci.yml — 逐行注释

```yaml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工作流名称，显示在 GitHub Actions 页面的标题
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
name: CI

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# "on" = 什么事件触发这个工作流
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
on:
  pull_request:           # 有人开 PR 时触发（包括更新 PR）
  push:
    branches:
      - main              # push 到 main 分支时触发
                           # 注意：push 到其他分支不会触发，
                           # 但 PR 里的 push 会通过上面的 pull_request 触发

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 作业（jobs）：可以并行运行的独立任务
# 这个工作流有两个作业：lint-type-test 和 build-wheel
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
jobs:

  # ── 作业 1：代码检查 + 类型检查 + 测试 ──────────────────
  lint-type-test:
    name: Lint, type-check, test (${{ matrix.python-version }})
    #                              ↑ 这是"表达式"语法，会被替换成实际的 Python 版本
    runs-on: ubuntu-latest
    #        ↑ 在 GitHub 提供的最新 Ubuntu 虚拟机上运行
    #          每次运行都是一台全新的机器，用完就扔

    # ── 矩阵策略：同一个作业用不同参数跑多次 ──
    strategy:
      fail-fast: false
      #          ↑ false = 即使 3.12 失败了，3.14 也继续跑
      #            true (默认) = 一个失败就全部取消
      matrix:
        python-version:
          - "3.12"
          - "3.14"
        # 结果：这个作业会跑两次，一次 3.12，一次 3.14
        # 在 GitHub Actions 页面上你会看到两行并排的结果

    # ── 步骤：按顺序执行 ──────────────────────────────────
    steps:

      # 步骤 1：把代码从 GitHub 拉到虚拟机上
      - name: Check out repository
        uses: actions/checkout@v7
        # "uses" = 调用别人写好的 Action（一个可复用的脚本包）
        # actions/checkout 是 GitHub 官方的，几乎每个工作流都用
        # @v7 = 用第 7 个大版本；新项目优先用当前稳定大版本

      # 步骤 2：安装 uv（Python 包管理器，比 pip 快很多）
      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
          # ↑ 缓存下载过的包，下次运行更快

      # 步骤 3：安装指定版本的 Python
      - name: Install Python
        run: uv python install ${{ matrix.python-version }}
        # "run" = 直接执行 shell 命令
        # ${{ matrix.python-version }} 会被替换成 "3.12" 或 "3.14"

      # 步骤 4：运行 Ruff（代码风格检查 / linter）
      - name: Ruff
        run: uv run --python ${{ matrix.python-version }} --no-project --with-editable ".[dev,diagnostics]" ruff check .
        # ruff check . = 检查当前目录下所有 Python 文件的代码风格
        # 如果有风格问题（未使用的导入、格式错误等），这一步会失败
        # --no-project --with-editable "." = 从本地源码安装 serie 包

      # 步骤 5：运行 Pyright（静态类型检查）
      - name: Pyright
        run: uv run --python ${{ matrix.python-version }} --no-project --with-editable ".[dev,diagnostics]" pyright
        # pyright 检查类型注解是否正确
        # 比如你写了 def f(x: int) 但传了一个 str，pyright 会报错

      # 步骤 6：运行 Pytest（单元测试）
      - name: Pytest
        if: matrix.python-version != '3.14'
        run: uv run --python ${{ matrix.python-version }} --no-project --with-editable ".[dev,diagnostics]" python -m pytest tests -q
        # 3.12 是兼容性测试：证明旧一些的 Python 也能跑。

      # 步骤 7：在主力 Python 版本上跑测试并输出覆盖率
      - name: Pytest with coverage
        if: matrix.python-version == '3.14'
        run: uv run --python ${{ matrix.python-version }} --no-project --with-editable ".[dev,diagnostics]" python -m pytest tests -q --cov --cov-report=term-missing --cov-report=xml
        # 3.14 是主力测试线：同样跑 pytest，但额外生成 coverage.xml。
        # 如果任何测试失败，整个 CI 会标红。

  # ── 作业 2：构建 wheel 并检查内容 ──────────────────────
  build-wheel:
    name: Build and inspect wheel
    runs-on: ubuntu-latest
    # 这个作业和 lint-type-test 并行运行（没有 needs: 依赖）

    steps:
      - name: Check out repository
        uses: actions/checkout@v7

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Install Python
        run: uv python install 3.14

      # 构建 wheel（.whl 文件）= pip install 时实际安装的包文件
      - name: Build distributions
        run: uv build . --out-dir dist
        # 产出：dist/serie-X.Y.Z-py3-none-any.whl
        # 这个 .whl 文件就是你 pip install serie 时下载的东西

      # 检查 wheel 里有什么，确保该有的有、不该有的没有
      - name: Inspect wheel contents
        run: |
          python - <<'PY'
          # ... (Python 脚本)
          # 这段代码打开 .whl 文件（其实是个 zip），检查：
          #   1. 必须包含 serie/demo_dist/index.html（HTML 模板）
          #   2. 不能包含 node_modules/、__pycache__/ 等垃圾
          # 如果不对，构建失败
          PY
```

**整体流程图：**

```
你 push 代码到 main
        │
        ▼
GitHub 检测到 push 事件，读取 .github/workflows/ci.yml
        │
        ├──▶ 作业 1: lint-type-test (Python 3.12)  ──▶ checkout → uv → ruff → pyright → pytest
        ├──▶ 作业 1: lint-type-test (Python 3.14)  ──▶ checkout → uv → ruff → pyright → pytest + coverage
        └──▶ 作业 2: build-wheel                    ──▶ checkout → uv → build → inspect wheel contents
        │
        ▼
全部绿色 ✅ = 代码没问题
任何红色 ❌ = 有问题，GitHub 会在 PR 或 commit 旁边显示
```

---

### publish.yml — 逐行注释

```yaml
name: Publish to PyPI

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 触发条件：两种方式
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
on:
  workflow_dispatch:       # 手动触发：在 GitHub Actions 页面点按钮
  push:
    tags:
      - "v*"               # 当你推送一个以 "v" 开头的 tag 时触发
                            # 比如：git tag v0.1.0 && git push --tags
                            # 这是发布新版本的标准流程

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 权限：这个工作流需要什么权限
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
permissions:
  contents: read           # 可以读代码
  id-token: write          # 可以生成 OIDC 令牌（用于 PyPI 可信发布）
                            # "可信发布" = PyPI 信任来自你的 GitHub 仓库的发布，
                            # 不需要手动管理 API key

jobs:
  publish:
    name: Build and publish
    runs-on: ubuntu-latest

    environment:
      name: pypi
      url: https://pypi.org/project/serie/
      # GitHub "环境" = 一组保护规则
      # 你可以在 GitHub 设置里要求发布前必须人工审批
      # 防止意外发布

    steps:
      - name: Check out repository
        uses: actions/checkout@v7

      - name: Install uv
        uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true

      - name: Install Python
        run: uv python install 3.14

      # 发布前再跑一遍完整检查（防止 tag 打在有问题的 commit 上）
      - name: Run release checks
        run: |
          uv run --python 3.14 --no-project --with-editable ".[dev,diagnostics]" ruff check .
          uv run --python 3.14 --no-project --with-editable ".[dev,diagnostics]" pyright
          uv run --python 3.14 --no-project --with-editable ".[dev,diagnostics]" python -m pytest tests -q

      # 构建 wheel
      - name: Build distributions
        run: uv build . --out-dir dist

      # 发布到 PyPI
      - name: Publish with PyPI Trusted Publishing
        run: uv publish --trusted-publishing always dist/*
        # --trusted-publishing always = 使用 OIDC 令牌（不需要 API key）
        # PyPI 验证"这个包是从 github.com/你的仓库 的 Actions 发出的"
        # 比存 API key 安全得多
```

**发布流程：**

```
你决定发布一个新版本
        │
        ▼
先把 pyproject.toml 里的 version 改成还没发布过的版本，例如 0.1.2
        │
        ▼
git tag v0.1.2 && git push --tags
        │
        ▼
GitHub 检测到 v* tag push，运行 publish.yml
        │
        ▼
ruff + pyright + pytest（最后一次检查）
        │
        ▼
uv build（生成 .whl）
        │
        ▼
uv publish → PyPI（全世界都能 pip install serie）
```

注意：PyPI 不允许重复上传同名文件。也就是说，如果 `serie-0.1.1.whl`
已经发过，就不能再发一次 0.1.1；必须先改版本号，例如 0.1.2，
再打 `v0.1.2` tag。

---

## Part 1.5: 可以直接在 YAML 里写学习注释吗？

可以。YAML 支持 `#` 注释，GitHub Actions 会忽略它们：

```yaml
# 这个 workflow 在 push 到 main 或开 PR 时运行
on:
  pull_request:
  push:
    branches:
      - main
```

但是我建议分两层：

1. **真实 `.github/workflows/*.yml` 保持简洁。**
   可以放少量关键注释，比如为什么用 PyPI Trusted Publishing，
   或为什么 wheel 检查要禁止 `node_modules/`。

2. **详细逐行学习版放在 `to_learn.md`。**
   真实 YAML 太多注释会影响阅读和维护；学习文档可以写得很长，
   可以解释每一行的概念、风险和替代方案。

所以最好的做法不是把真实 YAML 写成教材，而是：

```text
.github/workflows/ci.yml        # 干净、可维护、用于真实 CI
.github/workflows/publish.yml   # 干净、可维护、用于真实发布
to_learn.md                     # 本地学习版，详细解释，不上传
```

如果你之后想把学习材料公开，可以把 `to_learn.md` 精简成
`docs/maintainer-guide.md`；但现在它适合作为你的私有学习笔记。

---

## Part 2: 技能路线图

### Tier 1: 包发布与 CI（1-2 天/项）

已经有的：
- [x] GitHub Actions CI（ci.yml — 多版本测试矩阵）
- [x] PyPI 发布流水线（publish.yml — tag 触发 + 可信发布）

可以补的：

| 技能 | 做什么 | 简历写法 |
|------|--------|---------|
| **GitHub Pages 演示站** | CI 里 `serie demo lorenz` 生成 HTML，上传为 Pages artifact | "通过 GitHub Pages CI 流水线发布交互式研究演示" |
| **Pre-commit hooks** | ruff + pyright 在 commit 前本地跑，不用等 CI | "配置了 pre-commit 代码质量门禁" |
| **徽章（badges）** | README 里加 CI 状态、PyPI 版本、下载量徽章 | 视觉上让项目看起来"正式" |

### Tier 2: 容器化与可复现性（2-3 天/项）

| 技能 | 做什么 | 简历写法 |
|------|--------|---------|
| **Docker 镜像** | 一条 `docker run` 复现任何实验 | "用 Docker 容器化多语言（Python + Rust）可复现研究环境" |
| **Dev Container** | 任何人打开你的仓库就能在浏览器里运行 | "配置 GitHub Codespaces 实现零配置贡献者入职" |
| **确定性依赖锁定** | uv.lock 或 Nix flake 让依赖字节级一致 | "用确定性依赖解析实现可复现构建" |

### Tier 3: 云与分布式（3-7 天/项）

| 技能 | 做什么 | 简历写法 |
|------|--------|---------|
| **云端跑实验** | AWS/GCP 竞价实例，SSH 进去跑 `fern sweep` | "在云基础设施（AWS/GCP）上编排 GPU 实验运行" |
| **S3/GCS 产物存储** | 实验结果推到云存储而非本地磁盘 | "用云对象存储构建实验产物流水线" |
| **云端可查询实验 DB** | 你现有的实验 DB 放到云上，从任何地方查询 | "设计支持远程查询的实验追踪数据库" |
| **SERIE REST API** | FastAPI 包装：POST 数据源 + sigma，GET 分割结果 | "构建按需合成数据生成 REST API" |
| **并行实验扫描** | N 个噪声水平跨 N 个进程/机器并行 | "用 multiprocessing/Ray 实现并行实验扫描" |

### Tier 4: MLOps / 接近生产（1-2 周/项）

| 技能 | 做什么 | 简历写法 |
|------|--------|---------|
| **模型注册表** | 追踪哪个模型版本在哪些 SERIE 条件下测试过 | "构建模型认证注册表，关联预测器版本与压力测试结果" |
| **仪表盘部署** | fern_dash 吸引子查看器部署到一个 URL | "部署交互式数据可视化仪表盘（Fly.io/Railway）" |
| **监控与告警** | 定时实验失败或结果漂移时收到通知 | "实现实验健康监控与自动告警" |

### Tier 5: 系统 / 性能（持续）

| 技能 | 做什么 | 简历写法 |
|------|--------|---------|
| **Rust FFI / PyO3** | 从 Python 直接调用 fern_data | "用 PyO3 构建 Python-Rust FFI 桥接，数据管线加速 10-100x" |
| **WASM 构建** | 吸引子演示在浏览器原生运行，无需 Python | "将数值模拟编译为 WebAssembly 实现浏览器原生执行" |
| **基准测试套件** | 在 CI 中追踪性能回归 | "维护持续性能基准与回归检测" |

---

## Part 3: 关键概念速查

### Git 基础流程

```bash
# 日常开发
git add file.py           # 把改动放进"暂存区"（准备提交）
git commit -m "message"   # 把暂存区的改动打包成一个提交
git push                  # 把本地提交推送到 GitHub

# 发布版本
git tag v0.1.2            # 给当前 commit 打一个版本标签；版本号必须没发布过
git push --tags           # 把标签推送到 GitHub → 触发 publish.yml

# 分支
git checkout -b feature   # 创建并切换到新分支
git push -u origin feature # 推送新分支到 GitHub
# 然后在 GitHub 上开 Pull Request
```

### Nushell 替代方案

你的 shell 是 nushell，它比 bash 更适合日常开发——有类型、结构化数据、
管道返回表格而非文本。以下是常用操作的 nushell 写法：

```nu
# ── git 日常 ──────────────────────────────────────────────
git add file.py
git commit -m "message"
git push

# 发布版本
git tag v0.1.2
git push --tags

# 查看最近 5 个 commit（nushell 的 git log 返回结构化表格）
git log --oneline -5 | lines | each { |l| $l | str trim }

# 查看哪些文件改了（结构化输出）
git status --short | lines | parse "{status} {file}"
# 返回一个有 status 和 file 列的表格，比 bash 的纯文本好用得多

# ── 文件操作（nushell 比 bash 强很多的地方）────────────────
# 找所有 Python 文件
glob **/*.py | length                    # 文件计数
glob **/*.py | where ($it | path parse).stem != "__init__"  # 过滤

# 读 JSON（nushell 原生支持）
open manifest.json | get source_name     # 直接取字段
open manifest.json | get shape.clean     # 嵌套取值

# 读 CSV 为表格
open clean.csv | first 5                 # 前 5 行
open clean.csv | describe                # 列名和类型

# ── 实验相关 ──────────────────────────────────────────────
# 跑测试并只看失败的
python -m pytest tests -q 2>&1 | lines | where ($it | str contains "FAILED")

# 批量跑多个 sigma
[0.0 0.1 0.25 0.5 1.0] | each { |s|
    serie generate lorenz --sigma $s | from json | select sigma train_shape
}
# 返回一个表格，每行是一个 sigma 的结果

# 比较两个 JSON manifest
let a = (open tmp/run1/manifest.json)
let b = (open tmp/run2/manifest.json)
$a.shape == $b.shape  # true/false

# ── 为什么 nushell 比 bash 好 ─────────────────────────────
# bash: 所有东西都是字符串，用 grep/awk/sed 解析文本
# nushell: 数据有类型（表格、记录、列表），管道传结构化数据
#
# 例子：找出 serie/ 下超过 500 行的 Python 文件
# bash:   find serie -name "*.py" -exec wc -l {} + | sort -rn | awk '$1 > 500'
# nushell:
glob serie/**/*.py | each { |f| {file: $f, lines: (open $f | lines | length)} } | where lines > 500 | sort-by lines -r
```

**注意：GitHub Actions 的 `run:` 步骤默认用 bash，不能用 nushell。**
CI 里的命令必须是 bash 语法。nushell 是你本地开发的利器，但 CI YAML
里写的是给 GitHub 云端 Ubuntu 虚拟机执行的 bash。

如果你真的想在 CI 里用 nushell，可以加一个安装步骤：

```yaml
- name: Install Nushell
  run: cargo install nu --locked  # 但这很慢（~3 分钟编译）
- name: Run nu script
  run: nu my_script.nu
```

实际项目中很少这么做——CI 里的命令通常很简单，不值得为此装 nushell。

### YAML 语法速查

```yaml
# YAML 用缩进表示层级（类似 Python，但用空格不用 tab）
key: value                 # 键值对
list:                      # 列表
  - item1
  - item2
nested:                    # 嵌套
  child_key: child_value
multiline: |               # 多行字符串（保留换行）
  line 1
  line 2
expression: ${{ ... }}     # GitHub Actions 表达式（变量替换）
```

### CI/CD 术语

| 术语 | 意思 |
|------|------|
| CI (Continuous Integration) | 持续集成：每次 push 自动跑测试 |
| CD (Continuous Delivery) | 持续交付：测试通过后自动发布 |
| Workflow | 工作流：一个 .yml 文件定义的自动化流程 |
| Job | 作业：工作流里的一个独立任务，跑在自己的虚拟机上 |
| Step | 步骤：作业里按顺序执行的一条命令或动作 |
| Action | 动作：别人写好的可复用脚本（例如 `uses: actions/checkout@v7`） |
| Matrix | 矩阵：用不同参数（如多个 Python 版本）跑同一个作业 |
| Artifact | 产物：CI 过程中生成的文件（wheel、测试报告等） |
| Runner | 运行器：执行作业的虚拟机（GitHub 免费提供 Ubuntu/macOS/Windows） |
| Tag | 标签：给某个 commit 起个版本号名字（v0.1.0） |
| Wheel (.whl) | Python 包的安装格式（一个特殊的 zip 文件） |
| OIDC / 可信发布 | 不用密码/密钥，GitHub 和 PyPI 直接互信 |

### 建议学习顺序

1. **先理解 ci.yml**（你已经有了，本文档已解读）
2. **在本地试一次 git tag + push**，看 publish.yml 是否触发
3. **加一个 GitHub Pages 工作流**（下一个最容易的 CI 练习）
4. **写一个 Dockerfile**（学容器化的第一步）
5. **租一台云 VM 跑一次实验**（打破"只在本地跑"的边界）

---

## Part 4: 测试方法论——从「能跑」到「能信」

### 你现在有的测试层次

```
test_gen.py              → 积分器正确性（形状、有限性、混沌指标）
test_titration_edges.py  → 滴定层契约（确定性、注入恒等式、公共随机数）
test_contracts.py        → 显式行为契约（字段名、边界情况、别名安全、特征值恢复）
test_cli.py              → CLI 集成测试（每个子命令的 JSON 输出）
test_known_data_boundary.py → 已知数据错误消息
```

这已经比大多数研究项目好。但工业界还有几层你可能没听过的：

---

### 覆盖率测试（Coverage Testing）

**是什么：** 跑完所有测试后，统计代码里哪些行被执行过、哪些没有。

**为什么有用：** 你可能写了 100 个测试，但它们全都在测同一个函数。
覆盖率告诉你"你的测试根本没碰过 `center_and_spread()` 的 `robust_mad`
分支"——然后你补上。

**不是什么：** 覆盖率 100% 不等于没有 bug。一行代码被执行过不代表它的
输出被检查过。但覆盖率 < 50% 几乎一定意味着有大片代码完全没被测试过。

**怎么跑：**

```bash
# 安装
pip install pytest-cov

# 跑测试并生成覆盖率报告
python -m pytest tests -q --cov=serie --cov-report=term-missing

# 输出类似这样：
# Name                    Stmts   Miss  Cover   Missing
# serie/api.py              180     12    93%    145-148, 200-205
# serie/pipeline.py         320     45    86%    710-730, 890-920
# serie/catalog.py          280      0   100%
# TOTAL                     980     57    94%
#
# "Missing" 列告诉你哪些行从未被执行——那就是你需要补测试的地方
```

**在 CI 里加覆盖率门禁：**

```yaml
# ci.yml 里加一步
- name: Pytest with coverage
  run: |
    uv run ... python -m pytest tests -q --cov=serie --cov-fail-under=85
    # --cov-fail-under=85 = 覆盖率低于 85% 就让 CI 失败
```

**SERIE 具体例子：** 跑完覆盖率后你可能会发现 `pipeline.py` 里的
`StageDropBadColumns` 和 `StageImputeShortRuns` 覆盖率是 0%——因为
它们只在加载已知数据集（ETT、HuGaDB）时用到，而测试里没有真实数据文件。
这是一个合理的覆盖空白，但你应该知道它在哪。

---

### 属性测试（Property-Based Testing）

**是什么：** 你不写具体的输入输出对，而是声明一个"对所有合法输入都成立的性质"，
然后让框架自动生成几百个随机输入去验证。

**普通测试 vs 属性测试：**

```python
# ── 普通测试：你选一个具体输入，检查具体输出 ──
def test_sigma_zero_is_clean():
    bank = build_dgp_bank("lorenz", seed=0)
    split = inject_noise(bank, sigma=0.0)
    # 只测了 seed=0 这一个情况

# ── 属性测试：你声明一个性质，框架生成 100 个 seed 去验证 ──
from hypothesis import given, strategies as st

@given(seed=st.integers(0, 10000))
def test_sigma_zero_always_equals_clean(seed):
    bank = build_dgp_bank("lorenz", seed=seed)
    noisy = bank.build_observed_series(noise_level=0.0)
    np.testing.assert_array_equal(noisy, bank.clean[0])
    # hypothesis 会自动试 seed=0, 7832, 4091, ... 直到找到反例或跑完
```

**为什么有用：** 你写的普通测试只覆盖你想到的情况。属性测试覆盖你没想到的。
如果 hypothesis 发现 seed=8734 时你的函数崩溃了，它会自动"缩小"到最小的
反例（比如 seed=3）并告诉你。

**SERIE 可以测的属性：**

```python
from hypothesis import given, strategies as st, settings
import numpy as np
from serie import PartitionSpec, build_dgp_bank, inject_noise

# 属性 1：对任何合法的 sigma，注入是线性的
# observed = clean + sigma * draw，所以 (obs - clean) / sigma 应该等于 draw
@given(sigma=st.floats(min_value=0.01, max_value=100.0))
def test_injection_linearity_for_any_sigma(sigma):
    bank = build_dgp_bank("lorenz", seed=0)
    obs = bank.build_observed_series(noise_level=sigma)
    recovered_draw = (obs - bank.clean[0]) / sigma
    np.testing.assert_allclose(recovered_draw, bank.noise_draws[0], rtol=1e-3, atol=1e-4)


# 属性 2：对任何合法的分割比例，三个分割加起来 ≤ 总长度
@given(
    train_frac=st.floats(min_value=0.1, max_value=0.8),
    val_frac=st.floats(min_value=0.05, max_value=0.3),
)
@settings(max_examples=50)
def test_splits_never_exceed_total_length(train_frac, val_frac):
    test_start = train_frac + val_frac
    if test_start >= 0.99:
        return  # skip invalid combos
    p = PartitionSpec(
        train_start=0.0, train_end=train_frac,
        val_start=train_frac, val_end=test_start,
        test_start=test_start, test_end=1.0,
    )
    bank = build_dgp_bank("lorenz", seed=0)
    split = inject_noise(bank, sigma=0.0, partition=p)
    total = split.train_values.shape[0] + split.val_values.shape[0] + split.test_values.shape[0]
    assert total <= bank.T


# 属性 3：对任何 seed，clean 数组总是有限的（不含 NaN 或 inf）
@given(seed=st.integers(0, 10000))
@settings(max_examples=20)  # 每个 seed 要跑积分，所以限制数量
def test_lorenz_always_finite(seed):
    bank = build_dgp_bank("lorenz", seed=seed)
    assert np.all(np.isfinite(bank.clean))
```

**安装和运行：**

```bash
pip install hypothesis

# 跑属性测试
python -m pytest tests/test_properties.py -q
# hypothesis 默认每个 @given 跑 100 个例子
# 用 @settings(max_examples=N) 调整
```

**hypothesis 的杀手功能——缩小（shrinking）：**

如果它发现 seed=8734 导致测试失败，它不会直接报 8734。它会自动尝试
更小的值（8000? 100? 3?）直到找到最小的能触发 bug 的输入。
这让 debug 变得非常容易。

---

### 变异测试（Mutation Testing）

**是什么：** 自动修改你的源代码（比如把 `+` 改成 `-`、把 `>` 改成 `>=`），
然后重跑测试。如果测试还是通过了，说明你的测试没有检测到这个改动——
也就是说那行代码的正确性没有被真正验证过。

**和覆盖率的区别：**

```
覆盖率：这行代码被执行过了吗？（必要条件）
变异测试：如果这行代码错了，测试能发现吗？（充分条件）
```

**例子：**

```python
# 你的代码
def build_observed_series(self, noise_level):
    return self.clean[0] + noise_level * self.noise_draws[0]
#                          ↑ 变异测试会把这个 * 改成 +
#                            如果测试还通过了 = 你的测试太弱

# 变异体 1: clean + noise_level + noise_draws  (把 * 改成 +)
# 变异体 2: clean - noise_level * noise_draws  (把 + 改成 -)
# 变异体 3: clean + noise_level * noise_draws * 2  (改常数)
```

**怎么跑：**

```bash
pip install mutmut

# 对 serie/api.py 跑变异测试
mutmut run --paths-to-mutate=serie/api.py --tests-dir=tests/
# 输出：
#   Killed: 45   (测试检测到了变异 ✅)
#   Survived: 3  (测试没发现这 3 个变异 ⚠️)
#   变异分数: 93.75%

# 查看存活的变异体（= 你需要补的测试）
mutmut results
```

**实际操作：** 变异测试很慢（要跑 N × M 次测试，N = 变异体数，M = 测试数）。
通常只对核心模块跑（api.py、specs.py），不对整个仓库跑。

---

### 快照测试（Snapshot / Golden File Testing）

**是什么：** 第一次运行时保存输出到一个文件（"golden file"），之后每次运行
都跟这个文件比较。如果输出变了，测试失败——你必须明确选择"接受新输出"或"修 bug"。

**SERIE 场景：** 你可以把 `serie generate lorenz --sigma 0.1`
的输出存为 golden file。如果重构代码后输出的 JSON 结构变了，测试会抓住。

```python
# 用 pytest-snapshot 或 syrupy 库
def test_generate_lorenz_json_structure(snapshot):
    bank = build_dgp_bank("lorenz", seed=42)
    split = inject_noise(bank, sigma=0.1, partition=PartitionSpec.p70_20_10())
    result = {
        "train_shape": list(split.train_values.shape),
        "val_shape": list(split.val_values.shape),
        "test_shape": list(split.test_values.shape),
        "sigma": split.metadata["sigma"],
    }
    assert result == snapshot
    # 第一次跑：自动保存 result 到 __snapshots__/
    # 之后跑：和保存的 snapshot 比较
```

**更新 snapshot：** `pytest --snapshot-update`

---

### 模糊测试（Fuzz Testing）

**是什么：** 生成完全随机的、可能畸形的输入（乱码字符串、NaN 数组、
极端大小的值），看你的代码会不会崩溃。目的不是验证正确性，而是验证鲁棒性。

**和属性测试的区别：**

```
属性测试：我生成合法的随机输入，检查输出是否满足某个性质
模糊测试：我生成任意随机输入（包括非法的），检查代码是否崩溃
```

**SERIE 例子：** 如果有人传 `build_dgp_bank("lorenz", seed=NaN)` 或
`inject_noise(bank, sigma=float('inf'))` 会怎样？应该抛出一个清晰的
ValueError，而不是段错误或静默产生 NaN。

模糊测试在 Python 里不常用（Python 本身就不会段错误），
但对 Rust FFI（PyO3 桥接）非常有用——C/Rust 代码可以段错误。

---

### 术语对照表

| 术语 | 英文 | 你项目里的对应 |
|------|------|---------------|
| 单元测试 | Unit test | `test_lorenz_shape_and_finite` — 一个函数一个检查 |
| 集成测试 | Integration test | `test_cli_generate_writes_arrays` — 多个组件一起跑 |
| 契约测试 | Contract test | `test_noisy_equals_clean_plus_sigma_times_draw` — 验证数学恒等式 |
| 回归测试 | Regression test | 所有你的测试——防止改代码时弄坏已有功能 |
| 属性测试 | Property-based test | hypothesis — "对所有合法 seed 都成立" |
| 覆盖率 | Coverage | pytest-cov — "哪些行没被测试过" |
| 变异测试 | Mutation testing | mutmut — "如果代码错了，测试能发现吗" |
| 快照测试 | Snapshot / golden test | syrupy — "输出和上次一样吗" |
| 模糊测试 | Fuzz testing | 随机畸形输入，检查是否崩溃 |
| 测试矩阵 | Test matrix | CI 里 Python 3.12 + 3.14 并行跑 |
| 测试门禁 | Quality gate | `--cov-fail-under=85` — 低于阈值 CI 失败 |

### 建议优先级

1. **pytest-cov**（10 分钟）— 跑一次看哪些行没覆盖到
2. **hypothesis 属性测试**（1 小时）— 对注入线性性和分割完整性写 3-5 个
3. **快照测试**（30 分钟）— 对 CLI JSON 输出加 golden file
4. **变异测试**（可选）— 对 api.py 跑一次看变异分数
5. **模糊测试**（等 PyO3 桥接时再说）
