import atexit
import gc
import inspect
import logging
import os

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

supported_models = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Meta-Llama-3-70B',
]
supported_datasets = ['wikitext2', 'ptb', 'c4']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def config_logging(log_file, levels_to_log={logging.INFO, logging.ERROR}, to_console=True):
    """Dump the log both to console and a log file."""
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    class SpecificLevelFilter(logging.Filter):
        def filter(self, record):
            return record.levelno in levels_to_log

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.addFilter(SpecificLevelFilter())
    handlers = [file_handler]
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(SpecificLevelFilter())
        handlers.append(console_handler)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers)
    atexit.register(logging.shutdown)


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()
    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def distribute_model(model) -> None:
    """Distribute the model across available GPUs (offloading to CPU/disk if needed)."""
    no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(model, no_split_module_classes=no_split_module_classes)
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )
    cleanup_memory()
