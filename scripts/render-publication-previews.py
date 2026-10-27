"""Make silent, looping publication previews from the official project videos.

Requires ffmpeg; Python uses only the standard library. Sources are cached
outside the repository. Run: python scripts/render-publication-previews.py
"""
import argparse
from pathlib import Path
import subprocess
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'public/videos/publications'
CLIPS = {
    'humanoidttt': [
        ('ttt-wave', 'https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-03.mp4', 18.5, 4, (700, 0, 1080, 1080)),
        ('ttt-squat', 'https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-02.mp4', 20, 4.5, (700, 0, 1080, 1080)),
    ],
    'motionvla': [
        ('motion-turn-walk', 'https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt2_test1.mp4', 1, 4, (280, 0, 544, 544)),
        ('motion-walk-turn', 'https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt3_test1.mp4', 4, 3, (200, 0, 544, 544)),
    ],
}


def render(name, clips, cache):
    command = ['ffmpeg', '-v', 'error', '-y']
    filters = []
    for i, (key, url, start, duration, crop) in enumerate(clips):
        source = cache / (key + '.mp4')
        if not source.exists():
            with urlopen(url, timeout=60) as response:
                source.write_bytes(response.read())
        command += ['-ss', str(start), '-t', str(duration), '-i', str(source)]
        x, y, width, height = crop
        filters.append(f'[{i}:v]crop={width}:{height}:{x}:{y},scale=480:480:flags=lanczos,fps=20,setsar=1,setpts=PTS-STARTPTS[v{i}]')
    filters.append(''.join(f'[v{i}]' for i in range(len(clips))) + f'concat=n={len(clips)}:v=1:a=0[out]')
    target = OUTPUT / (name + '.mp4')
    command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '25', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-map_metadata', '-1', str(target)]
    subprocess.run(command, check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', '1.6', '-i', str(target), '-frames:v', '1', '-c:v', 'libwebp', '-quality', '85', str(OUTPUT / (name + '-poster.webp'))], check=True)
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
