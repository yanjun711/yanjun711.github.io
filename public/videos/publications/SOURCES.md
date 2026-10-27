# Publication previews

Silent excerpts from the author's official paper project pages. Original video
content is retained; editing consists of temporal selection, square crops,
resizing, and concatenation. No interpolated or generated frames are added.
Both previews are H.264 MP4, 480 × 480 at 20 fps, without an audio stream.

## HumanoidTTT

Project: https://aigeeksgroup.github.io/HumanoidTTT/

- Wave: https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-03.mp4
  — 18.5–22.5 s, crop (x=700, y=0, width=1080, height=1080).
- Squat: https://aigeeksgroup.github.io/HumanoidTTT/g1_realworld/HumanoidTTT-02.mp4
  — 20–24.5 s, same crop.

Combined duration: 8.5 s.

## MotionVLA

Project: https://aigeeksgroup.github.io/MotionVLA/

- Turn: https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt2_test1.mp4
  — 1–5 s, crop (x=280, y=0, width=544, height=544).
- Walk and turn: https://aigeeksgroup.github.io/MotionVLA/assets/video/demo/exo/exo_prompt3_test1.mp4
  — 4–7 s, crop (x=200, y=0, width=544, height=544).

Combined duration: 7 s.

## Regeneration

Run `python scripts/render-publication-previews.py` from the repository root.
Requires ffmpeg. Original videos are downloaded into a temporary cache rather
than committed. Static WebP posters are extracted at 1.6 s in each preview.
