"""
render_gif.py — render a match (evolved champion vs the 2015 baseline) to an
animated GIF, fully headless.

Uses the environment's pixel mode, which draws frames straight into numpy
arrays (no pyglet window needed). The policies are driven from the agents'
state observations read directly off the game object, so the same champion
that was trained on state observations can be filmed in pixel mode.

Usage:
  python render_gif.py results/ga_selfplay/ga_00111000.json --out results/figures/match.gif
"""

import argparse
import json

import cv2
import imageio.v2 as imageio
import numpy as np

import slimevolleygym.slimevolley as sv
from slimevolleygym.mlp import Model, games


def main():
    ap = argparse.ArgumentParser(description="Render champion vs baseline GIF")
    ap.add_argument("champion", help="champion .json checkpoint")
    ap.add_argument("--out", default="results/figures/match.gif")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--every", type=int, default=4,
                    help="capture every Nth physics step")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=float, default=12.5,
                    help="50/every keeps real-time speed")
    ap.add_argument("--seed", type=int, default=307)
    args = ap.parse_args()

    sv.setPixelObsMode()
    env = sv.SlimeVolleyEnv()
    env.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.champion) as f:
        params, streak = json.load(f)
    champ = Model(games["slimevolleylite"])
    champ.set_model_params(np.array(params))
    base = sv.BaselinePolicy()

    env.reset()
    frames = []
    score_r = score_l = 0
    t = 0
    for t in range(args.max_steps):
        # drive both policies from state observations read off the game object
        obs_r = env.game.agent_right.getObservation()
        obs_l = env.game.agent_left.getObservation()
        _, reward, done, _ = env.step(champ.predict(obs_r), base.predict(obs_l))
        if reward > 0:
            score_r += 1
        elif reward < 0:
            score_l += 1
        if t % args.every == 0:
            frame = env.render(mode="state")  # RGB numpy array, native res
            big = cv2.resize(
                frame,
                (frame.shape[1] * args.scale, frame.shape[0] * args.scale),
                interpolation=cv2.INTER_NEAREST)
            frames.append(np.clip(big, 0, 255).astype(np.uint8))
        if done:
            break

    imageio.mimsave(args.out, frames, duration=1.0 / args.fps, loop=0)
    print(f"{len(frames)} frames -> {args.out}")
    print(f"score (champion, right) vs (2015 baseline, left): "
          f"{score_r}-{score_l} in {t + 1} steps")


if __name__ == "__main__":
    main()
