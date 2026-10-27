"""Make silent, looping 2-by-2 publication preview grids from the official project videos.

Requires ffmpeg; Python uses only the standard library. Sources are cached
outside the repository. Run: python scripts/render-publication-previews.py
"""
import argparse
from pathlib import Path
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'public/videos/publications'
DURATION = 4
CLIPS = {
    'humanoidttt': [
        ('ttt-wave', 'https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-03.mp4', 18.5, (700, 0, 1080, 1080)),
        ('ttt-squat', 'https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-02.mp4', 20, (700, 0, 1080, 1080)),
        ('ttt-walk-turn', 'https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-01.mp4', 9, (200, 0, 1080, 1080)),
        ('ttt-wave-squat', 'https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-12.mp4', 10, (450, 0, 1080, 1080)),
    ],
    'motionvla': [
        ('motion-turn-walk', 'https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt2_test1.mp4', 1, (200, 0, 544, 544)),
        ('motion-walk-turn', 'https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt3_test1.mp4', 3.5, (160, 0, 544, 544)),
        ('motion-walk', 'https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt1_test1.mp4', 4.5, (260, 0, 544, 544)),
        ('motion-walk-2', 'https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt1_test2.mp4', 7, (230, 0, 544, 544)),
    ],
}


def render(name, clips, cache):
    command = ['ffmpeg', '-v', 'error', '-y']
    filters = []
    assert len(clips) == 4
    for i, (key, url, start, crop) in enumerate(clips):
        source = cache / (key + '.mp4')
        if not source.exists():
            with urlopen(url, timeout=60) as response:
                source.write_bytes(response.read())
        command += ['-ss', str(start), '-t', str(DURATION), '-i', str(source)]
        x, y, width, height = crop
        filters.append(f'[{i}:v]crop={width}:{height}:{x}:{y},scale=320:320:flags=lanczos,fps=20,setsar=1,setpts=PTS-STARTPTS[v{i}]')
    if name == 'motionvla':
        ego = cache / 'ego-turn-walk.mp4'
        if not ego.exists():
            with urlopen('https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/ego/ego_prompt2_test1.mp4', timeout=60) as response:
                ego.write_bytes(response.read())
        # This camera recording has a longer stationary lead-in. Select its
        # turning phase rather than equating the two files' recording clocks.
        command += ['-ss', '9', '-t', str(DURATION), '-i', str(ego)]
        filters += [
            '[4:v]scale=144:108:flags=lanczos,fps=20,setsar=1,setpts=PTS-STARTPTS,pad=148:112:2:2:color=white[ego]',
            '[v0][ego]overlay=4:4:shortest=1[pip]',
        ]
        tiles = ['pip', 'v1', 'v2', 'v3']
    else:
        tiles = [f'v{i}' for i in range(4)]
    filters.append(''.join(f'[{tile}]' for tile in tiles) + 'xstack=inputs=4:layout=0_0|320_0|0_320|320_320:shortest=1[out]')
    target = OUTPUT / (name + '-grid.mp4')
    command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '25', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-map_metadata', '-1', str(target)]
    subprocess.run(command, check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '1.6', '-i', str(target), '-frames:v', '1', '-c:v', 'libwebp', '-quality', '85', str(OUTPUT / (name + '-grid-poster.webp'))], check=True)
    print(name, f'{target.stat().st_size / 1024:.0f} KiB')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('/tmp/yanjun-publication-media'))
    args = parser.parse_args()
    args.source_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, clips in CLIPS.items():
        render(name, clips, args.source_dir)


if __name__ == '__main__':
    main()
