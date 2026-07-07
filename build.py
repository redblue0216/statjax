#!/usr/bin/env python3
"""StatJAX 构建脚本：使用 uv + pyproject.toml 打包 Python 项目

用法：
    python build.py                  # 打包 Python
    python build.py --install        # 打包 + 安装到当前环境
    python build.py --clean          # 先清理产物，再打包
    python build.py --verbose        # 显示详细输出

前置条件：
    - 已配置好 Python 环境（>= 3.10）
    - 项目根目录下包含 pyproject.toml 与 README.md
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _read_version() -> str:
    """Read version from pyproject.toml so build.py never goes stale."""
    pyproject = Path(__file__).parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else "unknown"


VERSION = _read_version()

PROJECT_ROOT = Path(__file__).parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def run(cmd: list[str], cwd: Path | None = None, verbose: bool = False) -> None:
    """执行命令，失败时退出"""
    if verbose:
        print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        print(f"  ❌ 命令失败: {' '.join(str(c) for c in cmd)}")
        if not verbose and result.stderr:
            print(result.stderr)
        sys.exit(1)


def check_prerequisites(verbose: bool) -> None:
    """检查前置条件"""
    print("【1/3】检查前置条件...")

    # 检查是否在项目根目录
    if not (PROJECT_ROOT / "pyproject.toml").exists():
        print(f"  ❌ 未找到 pyproject.toml，请确保在项目根目录下运行")
        sys.exit(1)

    if not (PROJECT_ROOT / "README.md").exists():
        print(f"  ❌ 未找到 README.md")
        sys.exit(1)

    # 检查 uv（不存在则自动安装）
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        uv_version = result.stdout.strip()
    except FileNotFoundError:
        print("  ⚠️  未找到 uv，正在安装...")
        run([sys.executable, "-m", "pip", "install", "uv"], verbose=verbose)
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        uv_version = result.stdout.strip()
    print(f"  ✅ {uv_version}")

    # 检查 Python
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  ✅ Python {py_version}")
    print(f"  ✅ 当前环境: {sys.prefix}")


def clean_build_artifacts(verbose: bool) -> None:
    """清理构建产物"""
    print("【0/3】清理构建产物...")
    targets = [BUILD_DIR, DIST_DIR, PROJECT_ROOT / "statjax.egg-info"]
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            if verbose:
                print(f"  已删除 {target}")
    # 清理源码中的 __pycache__，避免被打包进 wheel
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            if verbose:
                print(f"  已删除 {pycache}")
    print("  ✅ 清理完成")


def build_python(verbose: bool) -> None:
    """打包 Python"""
    print("【2/3】打包 Python...")
    print(f"  工作目录: {PROJECT_ROOT}")

    # uv build
    run(["uv", "build"], cwd=PROJECT_ROOT, verbose=verbose)

    # 验证产物
    wheels = list(DIST_DIR.glob("*.whl"))
    sdist = list(DIST_DIR.glob("*.tar.gz"))
    if not wheels:
        print("  ❌ 打包失败：未生成 wheel")
        sys.exit(1)

    print(f"  ✅ 打包完成")
    for whl in wheels:
        size_mb = whl.stat().st_size / (1024 * 1024)
        print(f"     {whl.name} ({size_mb:.1f} MB)")
    for tar in sdist:
        print(f"     {tar.name}")


def install_package(verbose: bool) -> None:
    """安装到当前环境"""
    print("【3/3】安装到当前环境...")
    wheels = sorted(DIST_DIR.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        print("  ❌ 未找到 wheel 文件")
        sys.exit(1)

    latest_wheel = wheels[0]
    print(f"  安装: {latest_wheel.name}")
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", str(latest_wheel)], verbose=verbose)

    # 验证
    result = subprocess.run(
        [sys.executable, "-c", "import statjax; print(statjax.__version__)"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✅ 安装成功，版本: {result.stdout.strip()}")
    else:
        print(f"  ⚠️  安装后验证失败")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"StatJAX {VERSION} 构建脚本：使用 uv 打包 Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  python build.py           # 打包
  python build.py --install # 打包 + 安装
  python build.py --clean   # 清理 + 打包
  python build.py --verbose # 显示详细输出
        """
    )
    parser.add_argument(
        "--install", "-i",
        action="store_true",
        help="打包后额外执行 pip install（安装到当前环境）"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="先清理 build/ dist/ egg-info/，再重新打包"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细命令输出"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  StatJAX {VERSION} 构建脚本")
    print(f"{'='*60}\n")

    start_time = time.time()

    # 清理
    if args.clean:
        clean_build_artifacts(args.verbose)

    # 前置检查
    check_prerequisites(args.verbose)

    # 打包 Python
    build_python(args.verbose)

    # 安装
    if args.install:
        install_package(args.verbose)
    else:
        print("【3/3】跳过安装（加 --install 自动安装）")

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  构建完成，耗时 {elapsed:.1f} 秒")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
