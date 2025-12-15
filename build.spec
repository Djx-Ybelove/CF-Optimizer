# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置文件 (spec文件)
作用：将 Cloudflare IP/域名优选工具 打包为 Windows 可执行文件(EXE)
环境：Python + PyInstaller，推荐搭配 UPX 压缩工具减小EXE体积
"""

# 加密相关的块密码对象，此处未启用加密，设为None
block_cipher = None

# ========================================================
# 资源文件映射配置（核心配置区）
# 格式说明：('本地源文件/文件夹路径', '打包后在EXE内部的对应路径')
# 作用：将外部资源文件嵌入到生成的EXE中，程序运行时可通过内部路径访问
# ========================================================
added_files = [
    # 1. 打包 IP/域名工具的核心资源文件夹
    #    本地路径：./official_ips_domain (包含colo.exe、locations.json、domain.txt等)
    #    内部路径：official_ips_domain (与代码中 self.ASSET_DIR / "official_ips_domain" 对应)
    # 2. 打包测速工具可执行文件
    #    本地路径：./cfst.exe (根目录的测速工具)
    #    内部路径：. (EXE内部根目录，与代码中 self.ASSET_DIR / self.EXE_CFST 对应)
    ('official_ips_domain', 'official_ips_domain'),
]

# ========================================================
# 分析阶段：扫描主程序的依赖项（模块、库、资源等）
# ========================================================
a = Analysis(
    ['cf_ip_domain_tool.py'],       # 主程序入口文件（核心业务逻辑文件）
    pathex=[],                      # 额外的模块搜索路径（此处无需配置，使用默认路径）
    binaries=[],                    # 需打包的二进制文件（已通过datas配置，此处留空）
    datas=added_files,              # 加载上面定义的资源文件映射配置
    hiddenimports=[],               # 隐式导入的模块（代码仅使用标准库，无需额外配置）
    hookspath=[],                   # 自定义钩子文件路径（默认即可）
    hooksconfig={},                 # 钩子配置参数（默认即可）
    runtime_hooks=[],               # 运行时钩子文件（默认即可）
    excludes=[],                    # 排除的模块（无需排除，保留所有依赖）
    win_no_prefer_redirects=False,  # Windows下不优先使用重定向（默认配置）
    win_private_assemblies=False,   # 不使用私有程序集（默认配置）
    cipher=block_cipher,            # 加密使用的块密码（此处为None，未加密）
    noarchive=False,                # 不将Python模块打包为归档文件（默认配置，便于调试）
)

# ========================================================
# 打包阶段：将分析后的纯Python代码打包为PYZ归档文件
# ========================================================
pyz = PYZ(
    a.pure,         # 纯Python模块（来自Analysis的结果）
    a.zipped_data,  # 压缩的数据（来自Analysis的结果）
    cipher=block_cipher,  # 加密配置（与上文一致）
)

# ========================================================
# 生成可执行文件：将所有依赖和资源打包为最终的EXE
# ========================================================
exe = EXE(
    pyz,                # 上述的PYZ归档文件
    a.scripts,          # 可执行的脚本（来自Analysis的结果）
    a.binaries,         # 二进制依赖文件（来自Analysis的结果）
    a.zipfiles,         # 压缩文件（来自Analysis的结果）
    a.datas,            # 资源文件（来自Analysis的datas配置）
    [],                 # 额外的资源参数（无）
    name='CF-Optimizer-Windows-AMD64',  # 生成的EXE文件名（可自定义）
    debug=False,        # 关闭调试模式（生产环境推荐False）
    bootloader_ignore_signals=False,  # 引导程序不忽略系统信号（默认配置）
    strip=False,        # 不剥离符号信息（Windows下无需配置）
    upx=True,           # 启用UPX压缩（需提前安装UPX，可显著减小EXE体积）
    upx_exclude=[],     # 排除UPX压缩的文件（无，全部压缩）
    runtime_tmpdir=None,  # 运行时临时目录（使用系统默认）
    console=True,       # 保留控制台窗口（关键：关闭则无法看到程序输出，需设为True）
    disable_windowed_traceback=False,  # 不禁用窗口模式的回溯（默认配置）
    argv_emulation=False,  # 禁用参数模拟（Windows下无需配置）
    target_arch=None,   # 目标架构（自动检测，支持x86/x64）
    codesign_identity=None,  # 代码签名标识（无，无需签名）
    entitlements_file=None,  # 权限文件（macOS下使用，Windows留空）
    icon=None,          # EXE图标文件（如需自定义，填写路径如'icon.ico'，此处默认）
)