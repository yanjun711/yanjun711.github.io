# Publication preview grids

Each preview spatially combines four real-robot videos from the author's official
project page in a 2 × 2 grid. All four clips start together, play simultaneously
for four seconds, and loop together. There are no sequential scene cuts.
Sources retain their original colors and playback speed, with square crops and
resizing. Output: silent H.264 MP4, 640 × 640, 20 fps, no audio stream.

## HumanoidTTT

Project: https://aigeeksgroup.github.io/HumanoidTTT/

| Tile | Source video | Start | Crop (x, y, width, height) |
| --- | --- | --- | --- |
| Top left | [HumanoidTTT-03.mp4](https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-03.mp4) | 18.5 s | 700, 0, 1080, 1080 |
| Top right | [HumanoidTTT-02.mp4](https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-02.mp4) | 20 s | 700, 0, 1080, 1080 |
| Bottom left | [HumanoidTTT-01.mp4](https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-01.mp4) | 9 s | 200, 0, 1080, 1080 |
| Bottom right | [HumanoidTTT-12.mp4](https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-12.mp4) | 10 s | 450, 0, 1080, 1080 |

## MotionVLA

Project: https://aigeeksgroup.github.io/MotionVLA/

| Tile | Source video | Start | Crop (x, y, width, height) |
| --- | --- | --- | --- |
| Top left | [exo_prompt2_test1.mp4](https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt2_test1.mp4) | 1 s | 200, 0, 544, 544 |
| Top right | [exo_prompt3_test1.mp4](https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt3_test1.mp4) | 3.5 s | 160, 0, 544, 544 |
| Bottom left | [exo_prompt1_test1.mp4](https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt1_test1.mp4) | 4.5 s | 260, 0, 544, 544 |
| Bottom right | [exo_prompt1_test2.mp4](https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt1_test2.mp4) | 7 s | 230, 0, 544, 544 |

The upper-left external-view demonstration includes a picture-in-picture from
its project-paired [ego camera video](https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/ego/ego_prompt2_test1.mp4).
The inset uses 9–13 s to select the corresponding turning phase after the ego
recording's longer stationary lead-in; raw recording timestamps are not shared.
It preserves the full 4:3 camera view, resized to 144 × 108 with a two-pixel
white edge. The inset sits at (4, 4), in background space beside the robot.
All five MotionVLA streams are encoded into the same four-second loop.

## Regeneration

Run `python scripts/render-publication-previews.py` from the repository root.
Requires ffmpeg. Original videos are cached outside the repository. The renderer
uses xstack to combine the four streams spatially, and extracts a grid poster
at 1.6 s. The page uses native muted autoplay and looping without playback
buttons or application-controlled pausing.
