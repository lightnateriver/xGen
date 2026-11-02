import pickle

import comfy.options

comfy.options.enable_args_parsing()

import os

os.environ["ACL_CONTEXT_MODE"] = "3"
import importlib.util
import folder_paths
import time
from comfy.cli_args import args
from app.logger import setup_logger
import itertools
import utils.extra_config
import logging
import sys
from comfy_execution.progress import get_progress_state
from comfy_execution.utils import get_executing_context
from comfy_api import feature_flags
from datetime import timedelta
from queue import Empty

if __name__ == "__main__":
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    import torch.distributed as dist


if __name__ == "__main__":
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)


def apply_custom_paths():
    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            utils.extra_config.load_extra_path_config(config_path)

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logging.info(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logging.info(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logging.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)


def execute_prestartup_script():
    if args.disable_all_custom_nodes and len(args.whitelist_custom_nodes) == 0:
        return

    def execute_script(script_path):
        module_name = os.path.splitext(script_path)[0]
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Failed to execute startup-script: {script_path} / {e}")
        return False

    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        node_prestartup_times = []

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = os.path.join(module_path, "prestartup_script.py")
            if os.path.exists(script_path):
                if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                    logging.info(
                        f"Prestartup Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                    continue
                time_before = time.perf_counter()
                success = execute_script(script_path)
                node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        logging.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            logging.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logging.info("")


apply_custom_paths()
execute_prestartup_script()

# Main code
import asyncio
import shutil
import threading
import gc

if os.name == "nt":
    os.environ['MIMALLOC_PURGE_DELAY'] = '0'

if __name__ == "__main__":
    if args.deterministic:
        if 'ACL_CONFIG' not in os.environ:
            os.environ['ACL_CONFIG'] = "mode=1"
            logging.info("Set ASCEND deterministic mode: ACL_CONFIG=mode=1")

if 'torch' in sys.modules:
    logging.warning(
        "WARNING: Potential Error in code: Torch already imported, torch should never be imported before this point.")
if __name__ == "__main__":
    import comfy.utils
    import execution
    import server
    from protocol import BinaryEventTypes
    import nodes
    import comfy.model_management
    import comfyui_version
    import app.logger
    import hook_breaker_ac10a0


# 主进程worker（在主线程执行，避免多线程问题）
def prompt_worker_main_thread(q, server_instance, rank, world_size):
    current_time: float = 0.0
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_none:
        cache_type = execution.CacheType.DEPENDENCY_AWARE

    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_size=args.cache_lru)

    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    # 进程同步点 - 确保所有从进程准备就绪
    dist.barrier()
    logging.info(f"主进程[{rank}] 所有从进程准备就绪，开始监听任务")

    while True:
        timeout = 1000.0
        if need_gc:
            timeout = max(gc_collect_interval - (current_time - last_gc_collect), 0.0)

        try:
            queue_item = q.get(timeout=timeout)
        except Empty:
            queue_item = None

        if queue_item is not None:
            item, item_id = queue_item
            execution_start_time = time.perf_counter()
            prompt_id = item[1]
            server_instance.last_prompt_id = prompt_id

            # 主进程广播任务（在主线程执行）

            broadcast_data = [item, item_id, True]

            dist.broadcast_object_list(broadcast_data, src=0)
            logging.info(f"主进程[{rank}] 广播任务成功 | ID: {item_id}")

            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True
            q.task_done(item_id,
                        e.history_result,
                        status=execution.PromptQueue.ExecutionStatus(
                            status_str='success' if e.success else 'error',
                            completed=e.success,
                            messages=e.status_messages))
            if server_instance.client_id is not None:
                server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id},
                                          server_instance.client_id)

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            if execution_time > 600:
                execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                logging.info(f"Prompt executed in {execution_time}")
            else:
                logging.info("Prompt executed in {:.2f} seconds".format(execution_time))

        # 处理内存释放和GC
        flags = q.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            need_gc = True
            last_gc_collect = 0

        if free_memory:
            e.reset()
            need_gc = True
            last_gc_collect = 0

        if need_gc:
            current_time = time.perf_counter()
            if (current_time - last_gc_collect) > gc_collect_interval:
                gc.collect()
                if torch.npu.is_available():
                    torch.npu.empty_cache()
                else:
                    comfy.model_management.soft_empty_cache()
                last_gc_collect = current_time
                need_gc = False
                hook_breaker_ac10a0.restore_functions()


# 从进程worker（在主线程执行，避免多线程问题）
def slave_worker_main_thread(prompt_server, rank, world_size):
    # 3. 绑定NPU设备（确保从进程使用正确的硬件）
    if torch.npu.is_available():
        torch.npu.set_device(rank)
        logging.info(f"从进程[{rank}] 绑定到NPU设备: {torch.npu.get_device_name(rank)}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(rank)
        logging.info(f"从进程[{rank}] 绑定到GPU设备: {torch.cuda.get_device_name(rank)}")

    # 4. 初始化模型管理器（控制内存/设备分配）

    logging.info(f"从进程[{rank}] 模型管理器初始化完成")

    # ---------------------- 原有逻辑：执行器初始化 ----------------------
    current_time: float = 0.0
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_none:
        cache_type = execution.CacheType.DEPENDENCY_AWARE

    # 初始化执行器（后端核心，依赖上述初始化）

    # ---------------------- 初始化执行器时传入伪server ----------------------
    e = execution.PromptExecutor(
        prompt_server,  # 传入包含client_id的伪server，而非None
        cache_type=cache_type,
        cache_size=args.cache_lru
    )

    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0

    logging.info(f"从进程[{rank}] 启动worker，等待主进程同步...")

    # 进程同步点 - 等待主进程准备就绪
    if world_size > 1:
        dist.barrier()
        logging.info(f"从进程[{rank}] 已与主进程同步，开始等待任务...")

    while True:
        try:
            # 从进程接收任务（在主线程执行）
            broadcast_data = [None, None, False]
            dist.broadcast_object_list(broadcast_data, src=0)

            logging.info(f"从进程[{rank}]接收成功")
            item, item_id, is_valid = broadcast_data

            if not is_valid:
                logging.info(f"从进程[{rank}] 收到退出信号，关闭worker...")
                comfy.model_management.unload_all_models()
                if torch.npu.is_available():
                    torch.npu.empty_cache()
                break

            if not item:
                time.sleep(0.1)
                continue

            logging.info(f"从进程[{rank}] 接收任务 | ID: {item_id}")
            execution_start_time = time.perf_counter()
            prompt_id = item[1]

            # ---------------------- 执行任务前的内存清理 ----------------------
            if torch.npu.is_available():
                torch.npu.empty_cache()

            # 执行任务（核心后端逻辑）
            e.execute(item[2], prompt_id, item[3], item[4])
            need_gc = True

            current_time = time.perf_counter()
            execution_time = current_time - execution_start_time
            logging.info(f"从进程[{rank}] 完成任务 | ID: {item_id}，耗时: {execution_time:.2f}s")

            # 处理内存释放和GC
            if need_gc:
                current_time = time.perf_counter()
                if (current_time - last_gc_collect) > gc_collect_interval:
                    gc.collect()
                    if torch.npu.is_available():
                        torch.npu.empty_cache()
                    else:
                        comfy.model_management.soft_empty_cache()
                    last_gc_collect = current_time
                    need_gc = False
                    hook_breaker_ac10a0.restore_functions()

        except Exception as e:
            logging.error(f"从进程[{rank}] 执行任务出错: {e}", exc_info=True)
            # 出错时强制清理资源
            comfy.model_management.unload_all_models()
            if torch.npu.is_available():
                torch.npu.empty_cache()
            gc.collect()
            time.sleep(1)


async def run(server_instance, address='', port=8188, verbose=True, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(
        server_instance.start_multi_address(addresses, call_on_start, verbose), server_instance.publish_loop()
    )


def hijack_progress(server_instance):
    def hook(value, total, preview_image, prompt_id=None, node_id=None):
        executing_context = get_executing_context()
        if prompt_id is None and executing_context is not None:
            prompt_id = executing_context.prompt_id
        if node_id is None and executing_context is not None:
            node_id = executing_context.node_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        if prompt_id is None:
            prompt_id = server_instance.last_prompt_id
        if node_id is None:
            node_id = server_instance.last_node_id
        progress = {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id}
        get_progress_state().update_progress(node_id, value, total, preview_image)

        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            if not feature_flags.supports_feature(
                    server_instance.sockets_metadata,
                    server_instance.client_id,
                    "supports_preview_metadata",
            ):
                server_instance.send_sync(
                    BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                    preview_image,
                    server_instance.client_id,
                )

    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    """
    清理临时目录，规范处理删除过程中的异常，避免静默忽略错误。
    """
    temp_dir = folder_paths.get_temp_directory()
    if not temp_dir:
        logging.warning("未获取到临时目录路径，跳过清理")
        return

    try:
        # 尝试删除临时目录（不使用ignore_errors，显式处理异常）
        shutil.rmtree(temp_dir)
        logging.info(f"临时目录清理成功: {temp_dir}")
    except FileNotFoundError:
        # 目录不存在，属于正常情况（可能已被其他进程清理）
        logging.info(f"临时目录不存在，无需清理: {temp_dir}")
    except PermissionError:
        # 权限不足，需提示用户检查权限
        logging.error(f"清理临时目录失败：权限不足，无法删除 {temp_dir}，请检查目录权限")
    except OSError as e:
        # 其他系统错误（如目录非空、文件被占用等）
        logging.error(f"清理临时目录失败：{e.strerror}，目录: {temp_dir}，错误码: {e.errno}")


def setup_database():
    try:
        from app.database.db import init_db, dependencies_available
        if dependencies_available():
            init_db()
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")


def start_comfyui(asyncio_loop=None, rank=0, world_size=1):
    # 仅主进程处理临时目录
    if rank == 0:
        if args.temp_directory:
            temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
            logging.info(f"Setting temp directory to: {temp_dir}")
            folder_paths.set_temp_directory(temp_dir)
        cleanup_temp()

    if args.windows_standalone_build and rank == 0:
        try:
            import new_updater
            new_updater.update_windows_updater()
        except (ImportError, AttributeError, OSError) as e:
            logging.warning(f"Windows updater skipped: {e}")

    # 仅主进程初始化前端服务
    prompt_server = None
    if rank == 0:
        if not asyncio_loop:
            asyncio_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(asyncio_loop)
        prompt_server = server.PromptServer(asyncio_loop)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        prompt_server = server.PromptServer(loop)

    # 初始化自定义节点
    hook_breaker_ac10a0.save_functions()
    if rank == 0:
        asyncio_loop.run_until_complete(nodes.init_extra_nodes(
            init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
            init_api_nodes=not args.disable_api_nodes
        ))
    else:
        loop.run_until_complete(nodes.init_extra_nodes(
            init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
            init_api_nodes=not args.disable_api_nodes
        ))

    hook_breaker_ac10a0.restore_functions()

    # 仅主进程初始化数据库和前端路由
    if rank == 0:
        setup_database()
        prompt_server.add_routes()
        hijack_progress(prompt_server)

    # 启动逻辑（核心修改：主线程执行worker）
    if rank == 0:
        # 前端服务用线程启动（不涉及HCCL通信）
        def start_frontend():
            asyncio_loop.run_until_complete(prompt_server.setup())
            asyncio_loop.run_until_complete(run(
                prompt_server,
                address=args.listen,
                port=args.port,
                verbose=not args.dont_print_server,
                call_on_start=call_on_start
            ))

        # 启动前端服务线程
        call_on_start = None
        if args.auto_launch:
            def startup_server(scheme, address, port):
                import webbrowser
                if os.name == 'nt' and address == '0.0.0.0':
                    address = '127.0.0.1'
                if ':' in address:
                    address = "[{}]".format(address)
                webbrowser.open(f"{scheme}://{address}:{port}")

            call_on_start = startup_server

        frontend_thread = threading.Thread(target=start_frontend, daemon=True)
        frontend_thread.start()

        # 主线程执行任务处理逻辑（包含HCCL通信）
        prompt_worker_main_thread(prompt_server.prompt_queue, prompt_server, rank, world_size)
    else:
        # 从进程主线程直接执行任务接收逻辑
        slave_worker_main_thread(prompt_server, rank, world_size)

    return None, None, None


if __name__ == "__main__":
    # 分布式初始化
    is_distributed = "RANK" in os.environ
    rank = 0
    world_size = 1
    timeout = timedelta(seconds=600)
    if is_distributed:
        try:
            # 优先选择HCCL后端（昇腾NPU）
            if torch.npu.is_available():
                backend = "hccl"
                device_type = "昇腾NPU"
            elif torch.cuda.is_available():
                backend = "nccl"
                device_type = "GPU"
            else:
                backend = "gloo"
                device_type = "CPU"

            logging.info(f"选择通信后端: {backend}（设备类型: {device_type}）")
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timeout,
            )
            rank = dist.get_rank()
            world_size = dist.get_world_size()

            # 绑定NPU设备
            if torch.npu.is_available():
                torch.npu.set_device(rank)
                device_name = torch.npu.get_device_name(rank)
            elif torch.cuda.is_available():
                torch.cuda.set_device(rank)
                device_name = torch.cuda.get_device_name(rank)
            else:
                device_name = "CPU"

            logging.info(f"分布式初始化完成 | 进程[{rank}/{world_size}] | 设备: {device_name}")
        except Exception as e:
            logging.error(f"分布式初始化失败: {e}", exc_info=True)
            sys.exit(1)

    # 打印版本信息
    if rank == 0:
        logging.info("Python version: {}".format(sys.version))
        logging.info("ComfyUI version: {}".format(comfyui_version.__version__))

    # 启动主从进程
    if rank == 0:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
        start_comfyui(asyncio_loop=event_loop, rank=rank, world_size=world_size)
    else:
        start_comfyui(rank=rank, world_size=world_size)
