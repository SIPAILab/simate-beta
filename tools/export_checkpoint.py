from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from sipai.engine.checkpoint import export_checkpoint
from sipai.utils import ensure_shared_artifact_directory, normalize_shared_artifact_permissions

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export SiPAI checkpoints to a standard Safetensors weight file.')
    parser.add_argument('--input', required=True, help='Input training checkpoint file or directory.')
    parser.add_argument('--output', required=True, help='Output path; the default Safetensors format requires a `.safetensors` filename.')
    parser.add_argument('--input-format', default='auto', choices=['auto', 'torch_full_state', 'full', 'state_dict', 'torch_dcp', 'pytorch_dcp', 'dcp', 'sharded_safetensors', 'safetensors'], help='Input checkpoint format. `auto` uses manifest.json and legacy format markers.')
    parser.add_argument('--output-format', default='safetensors', choices=['full', 'state_dict', 'wrapped_state_dict', 'safetensors'], help='Output payload format.')
    parser.add_argument('--dtype', default='fp32', choices=['fp32', 'bf16', 'fp16'], help='Floating-point dtype for Safetensors export.')
    parser.add_argument('--weights', default='ema', choices=['ema', 'raw'], help='Select EMA or raw model weights from the training checkpoint.')
    parser.add_argument('--config', default=None, help='Config YAML to include in a portable artifact.')
    parser.add_argument('--statistics', '--state-action-statistics', dest='statistics', default=None, help='State/action statistics JSON to copy beside the exported model.')
    parser.add_argument('--map-location', default='cpu', help='Torch map_location for full-state checkpoint input.')
    parser.add_argument('--no-portable-companions', action='store_true', help='Do not copy config.yaml or state_action_statistics.json beside the exported weights.')
    parser.add_argument('--summary-json', default=None, help='Optional JSON summary path.')
    return parser.parse_args()

def write_summary(summary: dict[str, object], output: str | Path | None) -> None:
    text = json.dumps(summary, indent=2, sort_keys=True) + '\n'
    if output:
        output_path = Path(output)
        ensure_shared_artifact_directory(output_path.parent)
        output_path.write_text(text, encoding='utf-8')
        normalize_shared_artifact_permissions(output_path)
    print(text, end='')

def main() -> None:
    args = parse_args()
    summary = export_checkpoint(args.input, args.output, checkpoint_format=args.input_format, output_format=args.output_format, map_location=args.map_location, metadata={'entrypoint': 'tools/export_checkpoint.py'}, export_dtype=getattr(args, 'dtype', 'fp32'), config_path=getattr(args, 'config', None), statistics_path=getattr(args, 'statistics', None), weights=getattr(args, 'weights', 'raw'), copy_portable_companions=not getattr(args, 'no_portable_companions', False))
    write_summary(summary, args.summary_json)
if __name__ == '__main__':
    main()
