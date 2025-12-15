import asyncio
import csv
import json
import platform
import random
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from contextlib import suppress, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Iterator, Optional, Iterable

# ==================== 🛠️ 1. 配置层 (Configuration) ====================
@dataclass
class AppConfig:
    """应用程序核心配置类，管理路径、参数及资源定位"""
    # 基础文件名常量
    EXE_COLO: str = "colo-windows-amd64.exe"  # IP扫描工具文件名
    EXE_CFST: str = "cfst.exe"  # 测速工具文件名
    URL_CFST: str = 'https://speed.cloudflare.com/__down?bytes=209715200'  # 测速文件URL
    FILE_COLO_CSV: str = "ip.csv"  # IP扫描结果CSV
    FILE_FINAL: str = "Official_bestip.csv"  # 最终输出文件
    FILE_LOC: str = "locations.json"  # 地区映射文件
    FILE_DOMAIN: str = "domain.txt"  # 域名列表文件
    FILE_CONFIG: str = "config.json"  # 配置文件
    
    # 默认业务参数
    PORTS: List[str] = field(default_factory=lambda: ["443", "2053", "2083", "2087", "2096", "8443"])  # 测试端口列表
    LIMITS: Dict[str, int] = field(default_factory=lambda: {  # 限制参数集
        "max_rounds": 1, "target_ip": 60, "colo_concurrency": 500, "top_n": 5, 
        "cfst_dn": 10, "domain_timeout": 5, "domain_min_latency": 40, 
        "domain_concurrency": 32, "domain_test_count": 4
    })
    DOMAIN_INTERVAL: float = 0.2  # 域名测试间隔(秒)
    IP_REGEX: re.Pattern = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')  # IP地址校验正则

    def __post_init__(self):
        """初始化路径解析，区分打包/未打包环境"""
        is_frozen = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
        self.BASE_DIR = Path(sys.executable).parent.resolve() if is_frozen else Path(__file__).parent.resolve()
        self.ASSET_DIR = Path(sys._MEIPASS).resolve() if is_frozen else self.BASE_DIR

        # 工作目录与文件路径
        self.dir_res = self.BASE_DIR / "result"  # 结果目录
        self.dir_task = self.BASE_DIR / "ips_country_port"  # 任务文件目录
        self.path_colo_csv = self.dir_res / self.FILE_COLO_CSV  # IP扫描结果路径
        self.path_final_out = self.BASE_DIR / self.FILE_FINAL  # 最终输出路径
        self.path_config_json = self.BASE_DIR / self.FILE_CONFIG  # 配置文件路径
        
        # 资源定位（优先外部，兼容内部）
        _asset_sub = self.ASSET_DIR / "official_ips_domain"
        self.path_colo_exe = _asset_sub / self.EXE_COLO  # IP扫描工具路径
        self.path_loc_json = _asset_sub / self.FILE_LOC  # 地区映射文件路径
        
        self.path_cfst_exe = self.ASSET_DIR / self.EXE_CFST  # 测速工具路径
        if not self.path_cfst_exe.exists():
            self.path_cfst_exe = _asset_sub / self.EXE_CFST

    @property
    def path_domain_txt(self) -> Path:
        """获取域名文件路径，优先使用用户目录下的文件"""
        user_file = self.BASE_DIR / self.FILE_DOMAIN
        if user_file.exists():
            print(f"  🔔 [提示] 已加载外部域名文件: {user_file.name}")
            return user_file
        return self.ASSET_DIR / "official_ips_domain" / self.FILE_DOMAIN

    def init_workspace(self):
        """初始化工作目录（创建不存在的目录）"""
        for p in (self.dir_task, self.dir_res):
            p.mkdir(parents=True, exist_ok=True)

    def load_external_config(self):
        """加载外部配置文件，不存在则生成默认配置"""
        if self.path_config_json.exists():
            try:
                print(f"  ⚙️  发现外部配置文件: {self.FILE_CONFIG}，正在加载...")
                data = json.loads(self.path_config_json.read_text(encoding='utf-8'))
                # 批量更新属性（字典类型合并，其他类型覆盖）
                for key, val in data.items():
                    if hasattr(self, key):
                        orig = getattr(self, key)
                        if isinstance(orig, dict) and isinstance(val, dict):
                            orig.update(val)
                        else:
                            setattr(self, key, val)
                print("  ✅ 配置加载成功！")
            except Exception as e:
                print(f"  ⚠️ 配置文件加载失败 ({e})，将使用默认参数。")
        else:
            try:
                export_data = {k: getattr(self, k) for k in ["PORTS", "URL_CFST", "DOMAIN_INTERVAL", "LIMITS"]}
                self.path_config_json.write_text(json.dumps(export_data, indent=4, ensure_ascii=False), encoding='utf-8')
                print(f"  ℹ️  已生成默认配置文件: {self.FILE_CONFIG} (您可以修改此文件来调整参数)")
            except Exception as e:
                print(f"  ⚠️ 无法生成配置文件: {e}")

# 全局配置实例
CONF = AppConfig()

# ==================== 🧱 2. 模型与工具 (Models & Utils) ====================
@dataclass(slots=True)
class ScanResult:
    """扫描结果数据模型，存储IP/域名的测试信息"""
    ip: str = ""  # IP地址
    port: str = ""  # 端口
    country: str = ""  # 地区码
    latency: float = 0.0  # 延迟(ms)
    speed: float = 0.0  # 速度(MB/s)
    loss: float = 0.0  # 丢包率(%)
    sent: int = 0  # 发送包数
    recv: int = 0  # 接收包数
    raw_domain: str = ""  # 原始域名

    def to_domain_line(self) -> str:
        """转换为域名结果字符串"""
        return f"{self.raw_domain}:{self.port}#CFD {self.latency:.2f}ms"

    def to_speed_line(self) -> str:
        """转换为IP速度结果字符串"""
        return f"{self.ip}:{self.port}#{self.country} {self.latency:.2f}ms {self.speed:.2f}MB/s"

class ConsoleUI:
    """控制台UI工具类，处理格式化输出"""
    @staticmethod
    def _pad(text: Any, width: int) -> str:
        """计算东亚字符宽度的填充，保证对齐"""
        s_text = str(text)
        # 东亚字符(全角)算2宽度，其他算1宽度
        v_len = sum(2 if unicodedata.east_asian_width(c) in 'FWA' else 1 for c in s_text)
        pad = max(0, width - v_len)
        return ' ' * (pad // 2) + s_text + ' ' * (pad - pad // 2)

    @staticmethod
    def separator(char: str = "=", length: int = 60):
        """打印分隔线"""
        print(char * length)

    @staticmethod
    def print_table(headers: List[Tuple[str, int]], rows: List[List[Any]]) -> None:
        """打印格式化表格"""
        if not rows: return
        total_w = sum(w for _, w in headers)
        ConsoleUI.separator("-", total_w)
        print("".join(ConsoleUI._pad(h, w) for h, w in headers))
        ConsoleUI.separator("-", total_w)
        for row in rows:
            print("".join(ConsoleUI._pad(cell, headers[i][1]) for i, cell in enumerate(row)))
        ConsoleUI.separator("-", total_w)

class SystemUtils:
    """系统工具类，处理文件、进程、命令执行等操作"""
    @staticmethod
    def clean_path(path: Path, is_dir: bool = False, pattern: str = "*"):
        """安全清理文件或目录（忽略错误）"""
        with suppress(OSError):
            if is_dir and path.exists():
                for item in path.glob(pattern):
                    if item.is_file(): item.unlink()
            elif not is_dir:
                path.unlink(missing_ok=True)

    @staticmethod
    def kill_processes(names: Iterable[str]):
        """终止指定进程（仅Windows）"""
        if platform.system() != "Windows": return
        # 过滤空值并构建命令
        targets = [n for n in names if n]
        if not targets: return
        
        cmd = ["taskkill", "/F"]
        for n in targets: cmd.extend(["/IM", n])
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def run_cmd_iter(cmd: List[str], cwd: Optional[Path] = None) -> Iterator[str]:
        """执行命令并生成实时输出流（行迭代）"""
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # 隐藏命令窗口
        
        try:
            with subprocess.Popen(
                cmd, cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                startupinfo=info, bufsize=1
            ) as p:
                if p.stdout:
                    yield from p.stdout
                p.wait()
                if p.returncode != 0: 
                    raise subprocess.CalledProcessError(p.returncode, cmd)
        except FileNotFoundError:
            raise RuntimeError(f"找不到执行文件: {cmd[0]}")

    @staticmethod
    def safe_rel_path(path: Path) -> str:
        """获取相对于程序根目录的安全路径（失败则返回绝对路径）"""
        try: return str(path.relative_to(CONF.BASE_DIR))
        except ValueError: return str(path)

    @staticmethod
    def iter_csv(path: Path) -> Iterator[List[str]]:
        """高效读取CSV文件（忽略错误行）"""
        if not path.exists(): return
        with suppress(OSError, UnicodeError):
            with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as f:
                # 过滤空行
                yield from csv.reader((line for line in f if line.strip()))

@contextmanager
def ProcessManager(proc_names: List[str]):
    """进程管理上下文管理器：进入前终止进程，退出后再次终止"""
    SystemUtils.kill_processes(proc_names)
    try: yield
    finally: SystemUtils.kill_processes(proc_names)

# ==================== 🌍 3. 核心业务 (Business Logic) ====================

# --- 模块 A: 域名优选 ---
async def fetch_domains() -> List[str]:
    """从本地文件读取域名列表（过滤空行和注释）"""
    f = CONF.path_domain_txt
    print(f"  正在从本地文件获取域名列表: {f.name}...")
    if not f.exists():
        print(f"  ❌ 错误: 找不到域名文件 {f}"); return []
    
    try:
        # 异步读取文件内容
        content = await asyncio.to_thread(f.read_text, encoding='utf-8', errors='ignore')
        # 提取有效域名（去重、排序）
        candidates = sorted({
            line.strip() for line in content.splitlines()
            if line.strip() and not line.startswith('#') and '.' in line
        })
        print(f"  ✅ 获取成功：{len(candidates)}个待测域名")
        return candidates
    except Exception as e:
        print(f"  ❌ 读取文件失败: {e}"); return []

async def test_single_domain(domain: str, sem: asyncio.Semaphore) -> ScanResult:
    """测试单个域名的延迟（带并发控制）"""
    async with sem:  # 控制并发数
        port = int(random.choice(CONF.PORTS))  # 随机选择测试端口
        latencies = []  # 存储成功的延迟数据
        limit, timeout = CONF.LIMITS["domain_test_count"], CONF.LIMITS["domain_timeout"]
        
        for _ in range(limit):
            start = time.perf_counter()
            try:
                # 建立连接测试延迟
                conn = asyncio.open_connection(domain, port)
                _, writer = await asyncio.wait_for(conn, timeout)
                latencies.append((time.perf_counter() - start) * 1000)  # 转换为ms
                writer.close()
                await writer.wait_closed()
            except (asyncio.TimeoutError, OSError):
                pass  # 忽略连接失败
            
            # 未达测试次数时等待间隔
            if len(latencies) < limit: await asyncio.sleep(CONF.DOMAIN_INTERVAL)
        
        count = len(latencies)
        avg_latency = sum(latencies) / count if count else 0.0  # 计算平均延迟
        return ScanResult(
            raw_domain=domain, port=str(port), sent=limit, recv=count,
            loss=(1 - count / limit) * 100, latency=avg_latency
        )

async def run_domain_test() -> List[str]:
    """执行域名TCPing延迟评估，返回优质域名列表"""
    ConsoleUI.separator(); print("🌐 [步骤 1/4]：域名 TCPing 延迟评估"); ConsoleUI.separator()
    candidates = await fetch_domains()
    if not candidates: return []

    print(f"  ⏳ 正在并发测试（{CONF.LIMITS['domain_concurrency']}线程）...")
    sem = asyncio.Semaphore(CONF.LIMITS["domain_concurrency"])  # 并发控制信号量
    
    # 批量测试所有域名
    results = await asyncio.gather(*(test_single_domain(d, sem) for d in candidates))
    # 按丢包率、延迟排序
    results.sort(key=lambda x: (x.loss, x.latency))

    print(f"\n  📋 域名测试结果:")
    rows = [[f"{r.raw_domain}:{r.port}", r.sent, r.recv, f"{r.loss:.2f}", f"{r.latency:.2f}"] for r in results]
    ConsoleUI.print_table([("域名", 36), ("已发送", 8), ("已接收", 8), ("丢包率", 8), ("平均延迟(ms)", 14)], rows)
    
    # 筛选前N条有效记录
    best = [r.to_domain_line() for r in results[:CONF.LIMITS["top_n"]] if r.recv > 0]
    print(f"\n  ✅ 测试完成：保存{len(best)}条优质记录（丢包率最低、延迟最优）\n")
    return best

# --- 模块 B: IP 扫描 ---
def parse_colo_results(csv_path: Path, loc_map: Dict[str, str], seen_ips: Dict[str, Set]) -> Dict[str, List[ScanResult]]:
    """解析IP扫描结果CSV，转换为按地区分组的ScanResult列表"""
    results = defaultdict(list)
    for row in SystemUtils.iter_csv(csv_path):
        if len(row) < 5: continue  # 过滤无效行
        ip, code, lat_raw = row[0], row[1], row[4]  # 提取IP、地区码、延迟字段
        
        country = loc_map.get(code)  # 转换为地区码
        # 过滤无效地区或已记录的IP
        if not country or ip in seen_ips[country]: continue
        
        # 提取延迟数值
        try:
            latency = int(''.join(filter(str.isdigit, lat_raw)))
            seen_ips[country].add(ip)  # 标记为已处理
            results[country].append(ScanResult(ip=ip, country=country, latency=latency))
        except ValueError:
            continue
    return results

async def run_ip_scan() -> Dict[str, List[ScanResult]]:
    """执行IP Colo扫描，返回按地区分组的IP列表（带延迟信息）"""
    if not CONF.path_colo_exe.exists():
        print(f"❌ 缺少核心工具：{CONF.EXE_COLO}"); return {}

    ConsoleUI.separator(); print("🌍 [步骤 2/4]：IP Colo 扫描"); ConsoleUI.separator()
    print(f"配置参数：共 {CONF.LIMITS['max_rounds']} 轮扫描 | 单地区目标：{CONF.LIMITS['target_ip']} 个 IP")

    # 加载地区映射表（IATA码→地区码）
    loc_map = {}
    if CONF.path_loc_json.exists():
        with suppress(Exception):
            txt = await asyncio.to_thread(CONF.path_loc_json.read_text, 'utf-8')
            loc_map = {i["iata"].upper(): i["cca2"].upper() for i in json.loads(txt) if "iata" in i}

    final_ips, seen_ips = defaultdict(list), defaultdict(set)  # 最终结果、已扫描IP记录
    
    for r in range(1, CONF.LIMITS["max_rounds"] + 1):
        print(f"\n  🔄 第 {r}/{CONF.LIMITS['max_rounds']} 轮扫描中...")
        
        # 提前终止条件：所有地区IP数量达标
        if r > 1 and final_ips and all(len(v) >= CONF.LIMITS["target_ip"] for v in final_ips.values()):
            print("    ✅ 检测到所有目标地区 IP 数量已达标，自动跳过后续轮次。"); break

        # 清理历史扫描结果
        SystemUtils.clean_path(CONF.path_colo_csv)
        # 构建扫描命令
        cmd = [str(CONF.path_colo_exe), "-ips", "4", "-task", str(CONF.LIMITS["colo_concurrency"]), "-outfile", str(CONF.path_colo_csv)]
        
        try:
            # 执行扫描命令并实时输出进度
            for line in SystemUtils.run_cmd_iter(cmd, cwd=CONF.path_colo_exe.parent):
                if "已完成" in line:
                    sys.stdout.write(f"\r    ⏳ 扫描进度: {line.strip()}"); sys.stdout.flush()
            print("")
        except RuntimeError as e:
            print(f"    ❌ 执行错误: {e}"); continue

        # 解析本轮扫描结果
        new_res = await asyncio.to_thread(parse_colo_results, CONF.path_colo_csv, loc_map, seen_ips)
        for cty, items in new_res.items():
            final_ips[cty].extend(items)

        # 统计各地区IP数量并截断到目标值
        stats = []
        for cty in final_ips:
            final_ips[cty].sort(key=lambda x: x.latency)  # 按延迟排序
            final_ips[cty] = final_ips[cty][:CONF.LIMITS["target_ip"]]  # 截断到目标数量
            stats.append((cty, len(final_ips[cty])))
        
        # 输出进度统计
        stats.sort(key=lambda x: x[1], reverse=True)
        top_stats = stats[:6]
        print(f"    📊 当前进度：{len(final_ips)}个地区 | 共{sum(x[1] for x in stats)}个IP")
        print(f"    🔝 地区：[{', '.join(f'{k}:{v}' for k, v in top_stats)} ...]")

    # 清理临时文件
    SystemUtils.clean_path(CONF.path_colo_csv)
    print("")
    return final_ips

# --- 模块 C: 速度测试 ---
def generate_speed_tasks(ip_data: Dict[str, List[ScanResult]], regions: Iterable[str]) -> List[Tuple[str, str, Path]]:
    """生成测速任务：按地区-端口分组IP，写入任务文件"""
    tasks = []
    if not CONF.PORTS: return []  # 无端口配置则返回空任务
    
    for cty in regions:
        ips = [r.ip for r in ip_data.get(cty, [])]  # 提取该地区所有IP
        if not ips: continue
        
        # 均匀分配IP到各端口
        chunk_size = (len(ips) + len(CONF.PORTS) - 1) // len(CONF.PORTS)  # 向上取整
        for i, port in enumerate(CONF.PORTS):
            sub_ips = ips[i * chunk_size : (i + 1) * chunk_size]  # 分片IP
            if sub_ips:
                t_file = CONF.dir_task / f"{cty}{port}.txt"  # 任务文件路径
                t_file.write_text('\n'.join(sub_ips), encoding='utf-8')  # 写入IP列表
                tasks.append((cty, port, t_file))
    return tasks

def parse_cfst_result(file_path: Path, cty: str, port: str) -> List[ScanResult]:
    """解析测速结果CSV，转换为ScanResult列表（按速度降序）"""
    res = []
    for row in SystemUtils.iter_csv(file_path):
        # 验证行格式和IP合法性
        if len(row) > 6 and CONF.IP_REGEX.match(row[0].strip()):
            try:
                res.append(ScanResult(
                    ip=row[0], port=port, country=cty,
                    sent=int(row[1]), recv=int(row[2]),
                    loss=float(row[3]), latency=float(row[4]),
                    speed=float(row[5])
                ))
            except (ValueError, IndexError):
                continue
    return sorted(res, key=lambda x: x.speed, reverse=True)  # 按速度降序

async def run_speed_test(ip_data: Dict[str, List[ScanResult]]) -> List[str]:
    """执行IP Cfst测速，返回优质IP列表"""
    if not ip_data: print("\n  ❌ 未扫描到有效IP，跳过测速步骤"); return []
    ConsoleUI.separator(); print("🎯 [步骤 3/4]：测速任务配置"); ConsoleUI.separator()
    
    # 按IP数量排序地区
    all_regions = sorted(ip_data.keys(), key=lambda k: len(ip_data[k]), reverse=True)
    print("  📍 可选地区列表：")
    for i in range(0, len(all_regions), 8):
        print(f"    {', '.join(f'{k}({len(ip_data[k])})' for k in all_regions[i:i+8])}")
        
    # 获取用户选择的测试地区
    print("\n  ⌨️  请选择测试地区：\n    - 测试全部: 回车\n    - 特定地区: 输入地区码(空格分隔)\n    - 跳过: 0")
    u_in = await asyncio.to_thread(input, "  📝 请输入您的选择 > ")
    if u_in.strip() == '0': return []

    # 处理用户输入
    sel_regions = set(all_regions)
    if u_in.strip():
        req = set(re.split(r'[,\s]+', u_in.strip().upper()))
        valid = req.intersection(ip_data.keys())
        if valid: sel_regions = valid
        else: print("  ⚠️ 输入无效，默认全部")

    # 显示选中的地区
    r_disp = str(sorted(list(sel_regions))[:10]) if len(sel_regions) <= 10 else f"[{len(sel_regions)} 个地区]"
    print(f"\n  ✅ 已锁定任务目标：{r_disp}\n  💾 正在生成测速任务文件... ")

    # 检查测速工具是否存在
    if not CONF.path_cfst_exe.exists():
        print(f"❌ 未找到{CONF.EXE_CFST}"); return []
    
    # 清理历史任务和结果
    SystemUtils.clean_path(CONF.dir_task, True)
    SystemUtils.clean_path(CONF.dir_res, True, "*.csv")
    
    ConsoleUI.separator(); print(f"⚡ 模块 4/4：IP Cfst 测速"); ConsoleUI.separator()
    # 生成测速任务
    tasks = await asyncio.to_thread(generate_speed_tasks, ip_data, sel_regions)
    print(f"  ▶️  任务队列：共 {len(tasks)} 个文件 \n")
    
    final_results = []
    # 串行测速（维持原设计）
    for idx, (cty, port, t_file) in enumerate(tasks):
        print(f"  --- ⏳ [{idx+1}/{len(tasks)}] 正在测试 {cty}{port} ----")
        o_file = CONF.dir_res / f"{cty}{port}.csv"  # 测速结果文件
        
        # 打印执行命令（相对路径）
        rel_exe = SystemUtils.safe_rel_path(CONF.path_cfst_exe).replace('/', '\\')
        rel_in = SystemUtils.safe_rel_path(t_file).replace('/', '\\')
        rel_out = SystemUtils.safe_rel_path(o_file).replace('/', '\\')
        print(f"  👉 执行命令: .\\{rel_exe} -tp {port} -f {rel_in} -url {CONF.URL_CFST} -dn {CONF.LIMITS['cfst_dn']} -p 0 -o {rel_out} \n")
        
        # 构建测速命令
        cmd = [str(CONF.path_cfst_exe), '-tp', port, '-f', str(t_file), '-url', CONF.URL_CFST, 
               '-dn', str(CONF.LIMITS['cfst_dn']), '-p', '0', '-o', str(o_file)]
        
        try:
            # 同步执行测速命令
            await asyncio.to_thread(subprocess.run, cmd, cwd=str(CONF.BASE_DIR), check=True)
            
            # 解析测速结果
            batch = await asyncio.to_thread(parse_cfst_result, o_file, cty, port)
            if batch:
                final_results.extend(batch)
                print(f"\n  📋 {cty}-{port} 测速结果：")
                rows = [[r.ip, r.sent, r.recv, f"{r.loss:.2f}", f"{r.latency:.2f}", f"{r.speed:.2f}", r.country] for r in batch]
                ConsoleUI.print_table([("IP 地址", 16), ("已发送", 8), ("已接收", 8), ("丢包率", 8), ("平均延迟(ms)", 14), ("下载速度(MB/s)", 16), ("地区码", 8)], rows)
                print("\n")
        except (subprocess.CalledProcessError, Exception) as e:
             print(f"  ❌ 任务执行中断或异常: {e}")

    # 聚合结果：按地区取前N名
    lines = []
    grouped = defaultdict(list)
    for r in final_results:
        grouped[r.country].append(r)
    
    for items in grouped.values():
        items.sort(key=lambda x: x.speed, reverse=True)  # 按速度排序
        lines.extend(r.to_speed_line() for r in items[:CONF.LIMITS["top_n"]])
    return lines

# ==================== 🚀 主程序 (Main) ====================
async def main():
    """主程序入口：初始化环境→执行各模块→输出结果"""
    CONF.init_workspace()  # 初始化工作目录
    print(""); ConsoleUI.separator()
    print("🚀 Cloudflare 综合优选工具 (All-in-One)\n🔍 执行流程：域名 TCPing 延迟评估 → IP Colo 扫描 → IP Cfst 测速")
    print(f"📁 最终结果将保存至：{CONF.FILE_FINAL}\n\n🔧 运行要求：\n - 系统：Windows (推荐)")
    print(f" - 依赖工具：已内置 {CONF.EXE_COLO} 和 {CONF.EXE_CFST}")
    print(" - 网络：清除系统代理，需要访问 Cloudflare\n\n❓ 用法：\n - 直接运行：自动执行所有模块")
    print(" - 自定义配置：修改同级目录下的 config.json")
    print(" - 自定义域名：在同级目录放置 domain.txt")
    print(" - 输入 '0' 在地区选择时跳过测速\n - Ctrl+C：安全中断并清理"); ConsoleUI.separator(); print("")

    print("正在初始化运行环境...")
    CONF.load_external_config()  # 加载配置文件
    
    # 进程管理：确保工具进程正确启动和终止
    with ProcessManager([CONF.EXE_COLO, CONF.EXE_CFST]):
        print("  [1/3] 🧹 清理历史残留进程...        ✅ 完成")
        print("  [2/3] 📂 重置任务与结果目录...      ✅ 完成")
        SystemUtils.clean_path(CONF.dir_task, True)
        SystemUtils.clean_path(CONF.dir_res, True)
        SystemUtils.clean_path(CONF.path_colo_csv)
        print("  [3/3] 🔍 检查内置依赖资源...        ✅ 就绪\n")

        # 执行各模块
        res_dom = await run_domain_test()  # 域名测试
        ip_map = await run_ip_scan()       # IP扫描
        res_spd = await run_speed_test(ip_map)  # 速度测试

        # 合并结果并输出
        final_data = res_dom + res_spd
        
        if not final_data and not ip_map:
             print("⚠️ 警告：未获得有效数据，跳过文件写入。")
        else:
            try:
                await asyncio.to_thread(CONF.path_final_out.write_text, "\n".join(final_data), encoding='utf-8')
                print(f"\n📂 最终文件已保存至：{CONF.FILE_FINAL}")
                
                if not final_data:
                    print("⚠️ 本次运行未产生任何有效结果")
                else:
                    print(f"💾 成功写入{len(final_data)}条数据（{len(res_dom)}条域名 + {len(res_spd)}条IP）")
                    if final_data:
                        print("\n🔍 数据预览:"); [print(f"    {l}") for l in final_data[:10]]
                        if len(final_data) > 5: print("    ....... \n")
            except Exception as e:
                print(f"❌ 写入结果文件失败: {e}")
                
        print("✅ 所有流程执行完毕！\n")

if __name__ == '__main__':
    # 适配Windows事件循环
    if platform.system() == "Windows": 
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\n🛑 用户中断，已安全退出。")
    except Exception as e:
        print(f"\n\n❌ 发生不可恢复错误: {e}\n")
        sys.exit(1)