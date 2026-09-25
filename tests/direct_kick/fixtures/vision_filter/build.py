import hashlib
import json
import argparse
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser(description='Regenerate the pinned C++ golden fixture in a temporary directory.')
parser.add_argument('--source-root', type=Path, default=Path('/home/akage/futbol_local/futbol_main'))
parser.add_argument('--output-dir', type=Path, default=Path('/tmp/direct_kick_vision_oracle'))
args = parser.parse_args()
ROOT = args.source_root
HERE = args.output_dir
HERE.mkdir(parents=True, exist_ok=True)
driver_path = Path(__file__).with_name('oracle.cpp')
if driver_path.resolve() != (HERE/'oracle.cpp').resolve():
    (HERE/'oracle.cpp').write_bytes(driver_path.read_bytes())
revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
if revision != '32ece6ee0676b1008d5bc58c3533d45613440568':
    raise SystemExit('Fixture source must be pinned revision 32ece6ee0676b1008d5bc58c3533d45613440568')
PKG = ROOT / 'src/ros2_ws/src/vision_filter'
ninja = (ROOT / 'src/ros2_ws/build/vision_filter/build.ninja').read_text()
section = ninja.split('build CMakeFiles/vision_filter_ball_prediction.dir/src/ball_prediction/cvkf.cpp.o:', 1)[1]
include_line = next(line for line in section.splitlines() if line.startswith('  INCLUDES = '))
includes = shlex.split(include_line.split(' = ', 1)[1])
sources = [PKG / 'src/ball_prediction/cvkf.cpp', PKG / 'src/ball_prediction/select_best_prediction.cpp']
command = ['g++', '-std=c++20', '-O2', *includes, str(HERE/'oracle.cpp'), *map(str,sources), '-o', str(HERE/'oracle')]
(HERE/'compile_command.json').write_text(json.dumps(command, indent=2)+'\n')
subprocess.run(command, check=True)
fixture = json.loads(subprocess.check_output([str(HERE/'oracle')], text=True))
paths = [*sources, PKG/'include/vision_filter/ball_prediction.hpp', PKG/'include/vision_filter/select_best_prediction.hpp', PKG/'src/vision_filter_node.cpp', PKG/'config/vision_filter.yaml']
fixture['provenance'] = {
    'repository': str(ROOT),
    'revision': subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
    'files_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    'driver_sha256': hashlib.sha256((HERE/'oracle.cpp').read_bytes()).hexdigest(),
    'driver_scope': 'Original CVKF and SelectBestPrediction classes; small node bank driver mirrors vision_filter_node.cpp:632-787. No ROS, localization synchronization, field gate, or detector confidence override.',
    'units': 'Input/output SI; original core computes in cm float64; log likelihood is original cm value.',
    'hypothesis_indices': ['stationary','rolling','high_speed','bounce'],
}
(HERE/'fixtures.json').write_text(json.dumps(fixture,indent=2,allow_nan=False)+'\n')
print('records:', {name: len(fixture[name]) for name in ['core_analytic','core','bank','node','interleaved']})
print('bank MAP selections:', [row['trace']['selected'] for row in fixture['bank']])
print('node events:', [(row['input']['stamp_ns'], row['event'],row['output']['status']) for row in fixture['node']])
