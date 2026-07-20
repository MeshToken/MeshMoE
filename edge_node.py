#!/usr/bin/env python3
"""
MeshMoE Edge Node Client v2
零配置: 自动检测硬件 → 选择模型 → 下载 → 注册 → 轮询任务 → 推理 → 赚积分
支持NAT穿透: 通过轮询Router获取任务，无需公网IP或端口映射
"""
import json, time, sys, os, socket, threading, signal, platform
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

# ============ 配置 ============
# 公网节点入口(nginx → router):注册/轮询/结果/chunk 全走这里,无需公网 IP
ROUTER_URL = os.getenv("MESHMOE_ROUTER_URL", "https://meshmoe.com/node")
MODEL_DIR = os.getenv("MESHMOE_MODEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
PORT = int(os.getenv("MESHMOE_PORT", "4002"))
NODE_ID = os.getenv("MESHMOE_NODE_ID", socket.gethostname())
POLL_INTERVAL = int(os.getenv("MESHMOE_POLL_INTERVAL", "5"))  # seconds between polls
N_CTX = int(os.getenv("MESHMOE_N_CTX", "4096"))
# ⭐ MOCK_MODE: 1=不加载真模型(返回固定响应),0=真模型推理
# 用途:验证 gateway→router→edge→计费→分成 控制链路(无 GPU/HF 连不上时)
MOCK_MODE = os.getenv("MOCK_MODE", "0") == "1"
MOCK_MODEL = os.getenv("MOCK_MODEL", "GLM-4-9B-Q4_K_M")   # mock 模式假装跑的模型
MOCK_TIER = os.getenv("MOCK_TIER", "standard")
MOCK_EXPERT = os.getenv("MOCK_EXPERT", "general")

# ⭐ 数据中心模式(2026-07-20):MESHMOE_INFER_URL 指向已有 OpenAI 兼容端点
# (vLLM / sglang / TGI / llama-server),edge_node 只做协议 worker,不下载/加载模型。
# 重型节点(H100/A100 集群)已有推理基建,接这个变量即入网。
INFER_URL = os.getenv("MESHMOE_INFER_URL", "").rstrip("/")
INFER_MODEL = os.getenv("MESHMOE_INFER_MODEL", "default")
# 远端端点的 key(vLLM 通常空;LiteLLM/one-api 系需要)
INFER_KEY = os.getenv("MESHMOE_INFER_KEY", "")
# ⭐ 自定义模型名 + 档位(INFER_URL 模式不受 MODEL_CATALOG 限制):
# MESHMOE_MODEL=DeepSeek-V3.2 MESHMOE_TIER=heavy 即可注册任意模型
MESHMOE_TIER = os.getenv("MESHMOE_TIER", "")
# ⭐ 账户绑定:注册带 api_key → 分成进自己账户(不填 = 匿名节点,分成无人认领)
MESHMOE_API_KEY = os.getenv("MESHMOE_API_KEY", "")

if MOCK_MODE:
    print("=" * 60)
    print("  MOCK MODE ENABLED - no real model loaded")
    print(f"  Pretending: {MOCK_MODEL} ({MOCK_TIER}/{MOCK_EXPERT})")
    print("  Set MOCK_MODE=0 for real inference (requires GPU + model)")
    print("=" * 60)

# ============ 模型库 ============
# 每个模型: name, gguf_file, size_mb, min_ram_gb, gpu_vram_mb, tier, source
# source: modelscope优先, huggingface备用
MODEL_CATALOG = {
    # --- Light: CPU / 核显, 2-4GB RAM ---
    "Qwen3-0.6B-Q8_0": {
        "gguf_file": "Qwen3-0.6B-Q8_0.gguf",
        "size_mb": 800,
        "min_ram_gb": 2,
        "gpu_vram_mb": 0,
        "tier": "light",
        "expert_type": "general",
        "description": "Qwen3 0.6B - 最小最快, CPU可跑(官方 Q8_0)",
        "modelscope_org": "Qwen",
        "modelscope_repo": "Qwen3-0.6B-GGUF",
        "huggingface_repo": "Qwen/Qwen3-0.6B-GGUF",
    },
    "Qwen2.5-1.5B-Instruct-Q4_K_M": {
        "gguf_file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_mb": 980,
        "min_ram_gb": 4,
        "gpu_vram_mb": 0,
        "tier": "light",
        "expert_type": "general",
        "description": "Qwen2.5 1.5B - 轻量但更聪明",
        "modelscope_org": "Qwen",
        "modelscope_repo": "Qwen2.5-1.5B-Instruct-GGUF",
        "huggingface_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    },
    # --- Standard: 6-12GB GPU ---
    "Qwen2.5-7B-Instruct-Q4_K_M": {
        "gguf_file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_mb": 4400,
        "min_ram_gb": 8,
        "gpu_vram_mb": 6000,
        "tier": "standard",
        "expert_type": "general",
        "description": "Qwen2.5 7B - 画质飞跃, 需6GB+显存",
        "modelscope_org": "Qwen",
        "modelscope_repo": "Qwen2.5-7B-Instruct-GGUF",
        "huggingface_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
    },
    "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M": {
        "gguf_file": "deepseek-r1-distill-qwen-7b-q4_k_m.gguf",
        "size_mb": 4400,
        "min_ram_gb": 8,
        "gpu_vram_mb": 6000,
        "tier": "standard",
        "expert_type": "reasoning",
        "description": "DeepSeek R1蒸馏 7B - 推理专精",
        "modelscope_org": "deepseek-ai",
        "modelscope_repo": "DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "huggingface_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF",
    },
    # --- Heavy: 16GB+ GPU ---
    "Qwen2.5-14B-Instruct-Q4_K_M": {
        "gguf_file": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "size_mb": 8700,
        "min_ram_gb": 16,
        "gpu_vram_mb": 16000,
        "tier": "heavy",
        "expert_type": "general",
        "description": "Qwen2.5 14B - 旗舰级, 需16GB+显存",
        "modelscope_org": "Qwen",
        "modelscope_repo": "Qwen2.5-14B-Instruct-GGUF",
        "huggingface_repo": "Qwen/Qwen2.5-14B-Instruct-GGUF",
    },
    "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M": {
        "gguf_file": "deepseek-r1-distill-qwen-14b-q4_k_m.gguf",
        "size_mb": 8700,
        "min_ram_gb": 16,
        "gpu_vram_mb": 16000,
        "tier": "heavy",
        "expert_type": "reasoning",
        "description": "DeepSeek R1蒸馏 14B - 推理之王",
        "modelscope_org": "deepseek-ai",
        "modelscope_repo": "DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "huggingface_repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    },
}


# ============ 硬件检测 ============
def detect_hardware():
    """自动检测硬件配置，返回推荐模型列表"""
    info = {
        "cpu_cores": os.cpu_count() or 4,
        "ram_gb": 0,
        "gpu_name": "",
        "gpu_vram_mb": 0,
        "platform": f"{platform.system()} {platform.machine()}",
    }

    # RAM
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        info["ram_gb"] = int(line.split()[1]) // 1048576
                        break
        elif platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            info["ram_gb"] = int(result.stdout.strip()) // 1073741824
    except:
        pass

    # GPU
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            info["gpu_name"] = parts[0].strip()
            vram_str = parts[1].strip().split()[0] if len(parts) > 1 else "0"
            info["gpu_vram_mb"] = int(float(vram_str))
    except:
        pass

    # 推荐模型
    recommended = []
    for name, cfg in MODEL_CATALOG.items():
        if cfg["gpu_vram_mb"] > 0 and info["gpu_vram_mb"] >= cfg["gpu_vram_mb"]:
            recommended.append(name)  # GPU够了
        elif cfg["gpu_vram_mb"] == 0 and info["ram_gb"] >= cfg["min_ram_gb"]:
            recommended.append(name)  # CPU模型, RAM够

    return info, recommended


# ============ 模型下载 ============
def download_model(model_name, model_dir):
    """从ModelScope下载模型，失败则尝试HuggingFace"""
    cfg = MODEL_CATALOG[model_name]
    gguf_file = cfg["gguf_file"]
    model_path = os.path.join(model_dir, gguf_file)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        print(f"[Download] Model already exists: {model_path}")
        return model_path

    os.makedirs(model_dir, exist_ok=True)

    # 尝试ModelScope
    ms_url = f"https://modelscope.cn/models/{cfg['modelscope_org']}/{cfg['modelscope_repo']}/resolve/main/{gguf_file}"
    # 尝试HuggingFace Mirror (国内可用)
    hf_url = f"https://hf-mirror.com/{cfg['huggingface_repo']}/resolve/main/{gguf_file}"
    # HuggingFace 直连(海外)
    hf_direct = f"https://huggingface.co/{cfg['huggingface_repo']}/resolve/main/{gguf_file}"

    for label, url in [("ModelScope", ms_url), ("HF-Mirror", hf_url), ("HuggingFace", hf_direct)]:
        print(f"[Download] Trying {label}: {url}")
        try:
            tmp_path = model_path + ".tmp"
            req = Request(url, headers={"User-Agent": "MeshMoE-Edge/2.0"})
            with urlopen(req, timeout=600) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded * 100 // total
                            size_mb = downloaded // 1048576
                            sys.stdout.write(f"\r[Download] {pct}% ({size_mb}MB)")
                            sys.stdout.flush()
            print()
            os.rename(tmp_path, model_path)
            print(f"[Download] Complete: {model_path}")
            return model_path
        except Exception as e:
            print(f"\n[Download] {label} failed: {e}")
            # Clean up partial download
            if os.path.exists(model_path + ".tmp"):
                os.remove(model_path + ".tmp")
            continue

    print(f"[Download] All sources failed. Manual download:")
    print(f"  wget -O {model_path} {ms_url}")
    print(f"  Or: wget -O {model_path} {hf_url}")
    return None


# ============ 推理引擎 ============
class EdgeNode:
    def __init__(self, model_name, model_path, expert_type="general"):
        self.model_name = model_name
        self.model_path = model_path
        self.expert_type = expert_type
        self.llm = None
        self.model_loaded = False
        self.tasks_processed = 0
        self.tokens_earned = 0
        self.running = True

    def load_model(self):
        if MOCK_MODE:
            print(f"[EdgeNode] MOCK MODE - skipping real model load")
            print(f"[EdgeNode] Pretending to run {self.model_name}")
            self.model_loaded = True
            return True
        if INFER_URL:
            # 数据中心模式:远端 OpenAI 兼容端点,健康检查后即就绪
            try:
                req = Request(f"{INFER_URL}/v1/models")
                with urlopen(req, timeout=10) as resp:
                    resp.read()
                print(f"[EdgeNode] Datacenter mode: remote endpoint OK ({INFER_URL})")
                self.model_loaded = True
                return True
            except Exception as e:
                print(f"[EdgeNode] Remote endpoint check failed: {e} (continuing anyway)")
                self.model_loaded = True
                return True
        try:
            from llama_cpp import Llama
        except ImportError:
            print("[EdgeNode] ERROR: llama-cpp-python not installed")
            print("[EdgeNode] Run: pip install llama-cpp-python")
            return False

        if not os.path.exists(self.model_path):
            print(f"[EdgeNode] ERROR: Model not found: {self.model_path}")
            return False

        print(f"[EdgeNode] Loading {self.model_name}...")
        n_threads = min(os.cpu_count() or 4, 8)

        # GPU offloading if available
        n_gpu_layers = 0
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                n_gpu_layers = -1  # offload all layers to GPU
                print("[EdgeNode] GPU detected, offloading all layers")
        except:
            pass

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=N_CTX,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        self.model_loaded = True
        print(f"[EdgeNode] Model loaded! Ready to process tasks.")
        return True

    def _remote_chat(self, messages, max_tokens, stream):
        """数据中心模式:打远端 OpenAI 兼容端点"""
        body = json.dumps({
            "model": INFER_MODEL, "messages": messages,
            "max_tokens": max_tokens, "temperature": 0.7, "stream": stream,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if INFER_KEY:
            headers["Authorization"] = f"Bearer {INFER_KEY}"
        req = Request(f"{INFER_URL}/v1/chat/completions", data=body,
                      headers=headers, method="POST")
        return urlopen(req, timeout=300)

    def infer_stream(self, messages, max_tokens=512):
        """E2-1 流式推理:生成器,yield (delta_text),最后一个 yield 后是 usage。
        用 create_chat_completion(官方 chat template,比手拼 prompt 诚实)。"""
        if MOCK_MODE:
            last_user = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user = msg.get("content", "")
                    break
            mock_text = f"[MOCK:{self.model_name}] Echo: {last_user[:80]} | replied at {time.strftime('%H:%M:%S')}"
            # 按词切片模拟流式
            words = mock_text.split(" ")
            for i, w in enumerate(words):
                yield ("delta", w + (" " if i < len(words) - 1 else ""))
                time.sleep(0.05)
            pt = len(last_user) // 4
            ct = min(max_tokens, max(8, len(mock_text) // 4))
            yield ("usage", {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct})
            return

        if not self.model_loaded:
            yield ("error", "model not loaded")
            return
        if INFER_URL:
            # 数据中心模式:SSE 透传解析
            try:
                resp = self._remote_chat(messages, max_tokens, True)
                buf = b""
                n = 0
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n\n" in buf:
                        frame, buf = buf.split(b"\n\n", 1)
                        for line in frame.decode("utf-8", errors="replace").splitlines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                continue
                            try:
                                v = json.loads(data)
                            except Exception:
                                continue
                            u = v.get("usage")
                            if u:
                                yield ("usage", u)
                            content = (v.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                            if content:
                                n += 1
                                yield ("delta", content)
                yield ("usage", {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n})
                self.tasks_processed += 1
                self.tokens_earned += n
            except Exception as e:
                yield ("error", str(e))
            return
        try:
            # llama.cpp 流式:每个含 content 的 part ≈ 1 token,直接计数(最诚实的 usage)
            n_completion = 0
            stream = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=0.7, stream=True)
            for part in stream:
                delta = part.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    n_completion += 1
                    yield ("delta", content)
                u = part.get("usage")
                if u:
                    yield ("usage", u)
            # 估算 prompt tokens(拼接消息过一遍 tokenizer,近似)
            try:
                concat = "\n".join(str(m.get("content", "")) for m in messages)
                n_prompt = len(self.llm.tokenize(concat.encode("utf-8")))
            except Exception:
                n_prompt = max(1, sum(len(str(m.get("content", ""))) for m in messages) // 4)
            yield ("usage", {
                "prompt_tokens": n_prompt,
                "completion_tokens": n_completion,
                "total_tokens": n_prompt + n_completion,
            })
            self.tasks_processed += 1
            self.tokens_earned += n_completion
        except Exception as e:
            yield ("error", str(e))

    def infer(self, messages, max_tokens=512):
        if MOCK_MODE:
            t0 = time.time()
            last_user = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user = msg.get("content", "")
                    break
            mock_text = f"[MOCK:{self.model_name}] Echo: {last_user[:80]} | replied at {time.strftime('%H:%M:%S')}"
            mock_tokens = min(max_tokens, max(8, len(mock_text) // 4))
            time.sleep(0.1)
            elapsed = time.time() - t0
            self.tasks_processed += 1
            self.tokens_earned += mock_tokens
            return {
                "choices": [{"message": {"role": "assistant", "content": mock_text}}],
                "model": f"meshmoe-edge/{self.model_name}",
                "usage": {"prompt_tokens": len(last_user) // 4, "completion_tokens": mock_tokens, "total_tokens": (len(last_user) // 4) + mock_tokens},
                "latency_ms": round(elapsed * 1000),
            }

        if not self.model_loaded:
            return {"error": "model not loaded"}

        t0 = time.time()
        if INFER_URL:
            # 数据中心模式:远端 OpenAI 兼容端点
            try:
                resp = self._remote_chat(messages, max_tokens, False)
                result = json.loads(resp.read())
                elapsed = time.time() - t0
                tokens = result.get("usage", {}).get("completion_tokens", 0)
                self.tasks_processed += 1
                self.tokens_earned += tokens
                result["latency_ms"] = round(elapsed * 1000)
                return result
            except Exception as e:
                return {"error": str(e), "latency_ms": round((time.time() - t0) * 1000)}
        try:
            # create_chat_completion:走模型官方 chat template(比手拼 prompt 诚实准确)
            result = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=0.7)
            elapsed = time.time() - t0
            text = result["choices"][0]["message"]["content"] if result.get("choices") else ""
            tokens = result.get("usage", {}).get("completion_tokens", 0)

            self.tasks_processed += 1
            self.tokens_earned += tokens

            return {
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "model": f"meshmoe-edge/{self.model_name}",
                "usage": result.get("usage", {}),
                "latency_ms": round(elapsed * 1000)
            }
        except Exception as e:
            elapsed = time.time() - t0
            return {"error": str(e), "latency_ms": round(elapsed * 1000)}


# ============ Router通信 ============
def submit_chunk(task_id, payload, latency_ms=0):
    """E2-1 边缘流式:向 Router 推一个 chunk。
    payload: {"delta": "..."} / {"finish": True, "usage": {...}} / {"error": "..."}"""
    data = json.dumps({
        "task_id": task_id,
        "peer_id": NODE_ID,
        "latency_ms": latency_ms,
        **payload,
    }).encode()
    for url in [f"{ROUTER_URL}/task/chunk", f"{ROUTER_URL}/api/task/chunk"]:
        try:
            req = Request(url, data=data, headers={
                "Content-Type": "application/json",
                "User-Agent": f"MeshMoE-Edge/2.0 ({NODE_ID})"
            }, method="POST")
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except:
            continue
    return None

def register_with_router(hardware_info, model_name, expert_type, tier):
    """向Router注册节点。MOCK_MODE 时自我申报 mock=true(诚实标注,L2 探针豁免)"""
    cfg = MODEL_CATALOG.get(model_name, {})
    data = json.dumps({
        "peer_id": NODE_ID,
        "expert_type": expert_type,
        "model_name": model_name,
        "port": PORT,
        "tier": tier,
        "hardware": hardware_info,
        "mock": MOCK_MODE,
        "api_key": MESHMOE_API_KEY,  # owner 绑定:分成到账(可空 = 匿名)
    }).encode()

    url = f"{ROUTER_URL}/api/nodes"  # Nginx proxies /api/nodes to Router
    # Also try direct Router URL
    for register_url in [url, f"{ROUTER_URL}/register"]:
        try:
            req = Request(register_url, data=data, headers={
                "Content-Type": "application/json",
                "User-Agent": f"MeshMoE-Edge/2.0 ({NODE_ID})",
                # 官方节点:MESHMOE_INTERNAL_KEY → verified(免 L2 探针/L3 抽检);
                # 第三方节点无 key → 必须过探针(防冒充设计)
                "X-Internal-Key": os.getenv("MESHMOE_INTERNAL_KEY", ""),
            }, method="POST")
            with urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                print(f"[Router] Registered: {result}")
                return True
        except Exception as e:
            continue

    print("[Router] Registration failed (will retry)")
    return False


def poll_task():
    """从Router轮询任务"""
    url = f"{ROUTER_URL}/task/poll/{NODE_ID}"
    # Also try via Nginx proxy
    for poll_url in [url, f"{ROUTER_URL}/api/task/poll/{NODE_ID}"]:
        try:
            req = Request(poll_url, headers={"User-Agent": f"MeshMoE-Edge/2.0 ({NODE_ID})"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except:
            continue
    return None


def submit_result(task_id, result, latency_ms, success=True):
    """向Router提交推理结果"""
    data = json.dumps({
        "task_id": task_id,
        "peer_id": NODE_ID,
        "result": result,
        "success": success,
        "latency_ms": latency_ms,
    }).encode()

    for url in [f"{ROUTER_URL}/task/result", f"{ROUTER_URL}/api/task/result"]:
        try:
            req = Request(url, data=data, headers={
                "Content-Type": "application/json",
                "User-Agent": f"MeshMoE-Edge/2.0 ({NODE_ID})"
            }, method="POST")
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except:
            continue
    return None


# ============ 本地健康检查 ============
class HealthHandler(BaseHTTPRequestHandler):
    """本地 :PORT/health 供用户自己查看节点状态"""
    def do_GET(self):
        if self.path == "/health":
            data = {
                "status": "ok",
                "node_id": NODE_ID,
                "model": edge.model_name if edge else "not loaded",
                "model_loaded": edge.model_loaded if edge else False,
                "tasks_processed": edge.tasks_processed if edge else 0,
                "tokens_earned": edge.tokens_earned if edge else 0,
                "uptime": round(time.time() - start_time),
            }
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args): pass


# ============ 主循环 ============
edge = None
start_time = time.time()

def worker_loop():
    """主工作循环: 轮询任务 → 推理 → 回报"""
    global edge

    # 注册(tier:目录优先,INFER_URL 自定义模型用 MESHMOE_TIER)
    hw_info, _ = detect_hardware()
    cfg = MODEL_CATALOG.get(edge.model_name, {})
    node_tier = cfg.get("tier") or MESHMOE_TIER or "light"
    node_expert = cfg.get("expert_type", "general")
    register_with_router(hw_info, edge.model_name, node_expert, node_tier)

    # 定时重新注册 (每5分钟)
    last_register = time.time()

    while edge.running:
        # 重新注册
        if time.time() - last_register > 300:
            register_with_router(hw_info, edge.model_name, node_expert, node_tier)
            last_register = time.time()

        # 轮询任务
        try:
            task = poll_task()
        except Exception as e:
            print(f"[Worker] Poll error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        if not task or task.get("status") == "idle":
            time.sleep(POLL_INTERVAL)
            continue

        # 有任务! 开始推理
        task_id = task.get("task_id", "")
        messages = task.get("messages", [])
        max_tokens = task.get("max_tokens", 512)
        is_stream = bool(task.get("stream"))

        print(f"[Worker] Task {task_id}{' (stream)' if is_stream else ''}: {messages[-1].get('content', '')[:60]}...")

        if is_stream:
            # E2-1 流式:边生成边推 chunk
            t0 = time.time()
            usage = {}
            error = None
            delta_buf = []
            last_flush = time.time()
            for kind, val in edge.infer_stream(messages, max_tokens):
                if kind == "delta":
                    delta_buf.append(val)
                    # 200ms 批量推一次,减少 HTTP  chatter
                    if time.time() - last_flush >= 0.2:
                        submit_chunk(task_id, {"delta": "".join(delta_buf)})
                        delta_buf = []
                        last_flush = time.time()
                elif kind == "usage":
                    usage = val
                elif kind == "error":
                    error = val
                    break
            if delta_buf:
                submit_chunk(task_id, {"delta": "".join(delta_buf)})
            latency = round((time.time() - t0) * 1000)
            if error:
                submit_chunk(task_id, {"error": error}, latency)
                print(f"[Worker] Stream task {task_id} failed: {error}")
            else:
                if not usage:
                    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                submit_chunk(task_id, {"finish": True, "usage": usage}, latency)
                print(f"[Worker] Stream task {task_id} done ({latency}ms)")
        else:
            result = edge.infer(messages, max_tokens)
            latency = result.get("latency_ms", 0)
            success = "error" not in result

            # 提交结果
            submit_result(task_id, result, latency, success)

            if success:
                print(f"[Worker] Task {task_id} done ({latency}ms, {result.get('usage',{}).get('completion_tokens',0)} tokens)")
            else:
                print(f"[Worker] Task {task_id} failed: {result.get('error', 'unknown')}")

        # 短暂休息避免过热
        time.sleep(1)


def main():
    global edge, start_time
    start_time = time.time()

    print("=" * 50)
    print("  MeshMoE Edge Node v2")
    print("  Share compute, earn credits")
    print("=" * 50)

    # 1. 检测硬件
    hw_info, recommended = detect_hardware()
    print(f"\n[Hardware] CPU: {hw_info['cpu_cores']} cores")
    print(f"[Hardware] RAM: {hw_info['ram_gb']}GB")
    if hw_info['gpu_name']:
        print(f"[Hardware] GPU: {hw_info['gpu_name']} ({hw_info['gpu_vram_mb']}MB VRAM)")
    else:
        print("[Hardware] GPU: None (CPU-only mode)")

    # 2. 选择模型
    # 优先用环境变量指定
    chosen = os.getenv("MESHMOE_MODEL")

    if MOCK_MODE:
        chosen = MOCK_MODEL
        print(f"\n[Model] MOCK MODE - using: {chosen} (no real download/load)")
    elif not chosen:
        if recommended:
            chosen = recommended[0]
            print(f"\n[Model] Auto-selected: {chosen}")
            print(f"  {MODEL_CATALOG[chosen]['description']}")
        else:
            chosen = "Qwen3-0.6B-Q8_0"
            print(f"\n[Model] Defaulting to smallest: {chosen}")

    if MOCK_MODE:
        cfg = {"tier": MOCK_TIER, "expert_type": MOCK_EXPERT, "description": "MOCK - no real model", "size_mb": 0}
        model_path = "/tmp/mock_model.gguf"
    elif INFER_URL:
        # 数据中心模式:任意模型名,档位用 MESHMOE_TIER(默认 heavy)
        cfg = {
            "tier": MESHMOE_TIER or "heavy",
            "expert_type": "general",
            "description": f"remote endpoint ({INFER_URL})",
            "size_mb": 0,
        }
        model_path = None
        if chosen not in MODEL_CATALOG:
            print(f"[Model] Custom datacenter model: {chosen} (tier={cfg['tier']}, not in local catalog — OK for INFER_URL mode)")
    else:
        if chosen not in MODEL_CATALOG:
            print(f"[Model] Unknown model: {chosen}")
            print(f"[Model] Available: {', '.join(MODEL_CATALOG.keys())}")
            sys.exit(1)

        cfg = MODEL_CATALOG[chosen]
        print(f"[Model] {chosen} ({cfg['description']})")
        print(f"[Model] Tier: {cfg['tier']}, Expert: {cfg['expert_type']}, Size: ~{cfg['size_mb']}MB")

        # 3. 下载模型
        model_path = download_model(chosen, MODEL_DIR)
        if not model_path:
            sys.exit(1)

    # 4. 加载模型(mock 模式下 load_model 内部跳过)
    edge = EdgeNode(chosen, model_path, cfg.get("expert_type", "general"))
    if not edge.load_model():
        sys.exit(1)

    # 5. 启动本地健康检查服务
    try:
        health_server = HTTPServer(("127.0.0.1", PORT), HealthHandler)
        health_server.allow_reuse_address = True
        threading.Thread(target=health_server.serve_forever, daemon=True).start()
        print(f"[Health] Local status at http://127.0.0.1:{PORT}/health")
    except Exception as e:
        print(f"[Health] Could not start health server: {e}")

    # 6. 启动工作循环
    print(f"\n[Worker] Starting... Polling {ROUTER_URL} for tasks")
    print(f"[Worker] Node ID: {NODE_ID}")

    worker_thread = threading.Thread(target=worker_loop, daemon=True)
    worker_thread.start()

    # 7. 主线程等待退出信号
    def shutdown(sig, frame):
        print(f"\n[MeshMoE] Shutting down...")
        edge.running = False
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        while edge.running:
            time.sleep(1)
            # 打印状态
            if edge.tasks_processed > 0 and int(time.time()) % 60 == 0:
                uptime_min = int((time.time() - start_time) / 60)
                print(f"[Status] Uptime: {uptime_min}min | Tasks: {edge.tasks_processed} | Tokens: {edge.tokens_earned}")
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
