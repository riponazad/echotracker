"""
Interactive EchoTracker Demo — Single Window
Plays video, lets you select tracking points, runs EchoTracker and TAPIR inference,
and shows results side-by-side in the same window.

Usage: python demo2.py <video_name>
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
import cv2 as cv
import torch
import threading

from model.net import EchoTracker, TAPIR
from utils.utils_ import paint_vid, add_text_to_frames
from utils import viz_utils
import mediapy as media


class EchoTrackerDemo:

    def __init__(self, video_path):
        self.video_path = video_path
        self.display_w, self.display_h = 640, 480
        self.colormap = viz_utils.get_colors(40)

        # state machine: loading | playing | selecting | tracking | results
        self.state = 'loading'
        self.select_points = []
        self.play_idx = 0
        self.result_idx = 0
        self.echo_frames = None    # (T, H, W, 3) — kept separate to preserve aspect ratio
        self.tapir_frames = None   # (T, H, W, 3)
        self._tracking_done = False
        self._models_ready = False
        self.ax2 = None   # second axes, only visible during results
        self.im2 = None

        self.frames_gray = self._load_video()   # (T, H, W) uint8
        self.T = len(self.frames_gray)

        self._build_ui()

        # Load models in background so the UI stays responsive
        threading.Thread(target=self._load_models, daemon=True).start()

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def _load_video(self):
        cap = cv.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.resize(frame, (self.display_w, self.display_h))
            frames.append(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        cap.release()
        if not frames:
            raise ValueError("Video contains no frames.")
        return np.array(frames)

    # ------------------------------------------------------------------
    # Model loading (background thread)
    # ------------------------------------------------------------------

    def _load_models(self):
        self.echotracker = EchoTracker(device_ids=[0])
        self.echotracker.load(path="model/weights/echotracker", eval=True)
        self.tapir = TAPIR(pyramid_level=0, device_ids=[0], ft_model=True)
        self.tapir.load(path="model/weights/tapir/finetuned")
        self._models_ready = True

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('#1a1a2e')
        try:
            self.fig.canvas.manager.set_window_title('EchoTracker — Interactive Demo')
        except Exception:
            pass

        # Main image axes (occupies most of the window)
        self.ax = self.fig.add_axes([0.01, 0.12, 0.98, 0.86])
        self.ax.set_facecolor('black')
        self.ax.axis('off')
        rgb0 = cv.cvtColor(self.frames_gray[0], cv.COLOR_GRAY2RGB)
        self.im = self.ax.imshow(rgb0, aspect='auto')
        self.ax.set_title('Loading models — please wait…', color='white', pad=5)

        # Status and point-count text below the image
        self.status_txt = self.fig.text(
            0.5, 0.075, 'Initializing…',
            ha='center', va='center', fontsize=10, color='#ccccff',
        )
        self.pts_txt = self.fig.text(
            0.5, 0.048, '',
            ha='center', va='center', fontsize=9, color='#ffdd88',
        )

        # Three buttons across the bottom
        btn_kw = dict(color='#2a2a4a', hovercolor='#44446a')
        self.btn_select = Button(
            self.fig.add_axes([0.04, 0.005, 0.28, 0.038]),
            'Pause & Select Points', **btn_kw)
        self.btn_clear = Button(
            self.fig.add_axes([0.36, 0.005, 0.28, 0.038]),
            'Clear Points', **btn_kw)
        self.btn_track = Button(
            self.fig.add_axes([0.68, 0.005, 0.28, 0.038]),
            'Run Tracking', **btn_kw)
        for b in (self.btn_select, self.btn_clear, self.btn_track):
            b.label.set_color('white')

        self.btn_select.on_clicked(self._on_pause_select)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_track.on_clicked(self._on_run_tracking)

        # Animation — always running; state drives what gets drawn
        self.ani = FuncAnimation(
            self.fig, self._tick, interval=50, blit=False, cache_frame_data=False,
        )

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    # ------------------------------------------------------------------
    # Animation tick — main thread, called ~20 fps
    # ------------------------------------------------------------------

    def _tick(self, _):
        # Transition: models just finished loading
        if self.state == 'loading' and self._models_ready:
            self.state = 'playing'
            self.ax.set_title('', color='white')
            self.status_txt.set_text(
                'Video playing.  Click "Pause & Select Points" to choose tracking targets.'
            )

        # Transition: tracking finished in background thread
        if self._tracking_done:
            self._tracking_done = False
            self.state = 'results'
            self.result_idx = 0
            self._switch_to_dual(self.echo_frames[0], self.tapir_frames[0])
            self.status_txt.set_text(
                'Tracking complete!  Click "Pause & Select Points" to try different points.'
            )
            self.pts_txt.set_text('')

        # Update the displayed frame according to current state
        if self.state == 'playing':
            self.play_idx = (self.play_idx + 1) % self.T
            self.im.set_data(cv.cvtColor(self.frames_gray[self.play_idx], cv.COLOR_GRAY2RGB))
        elif self.state == 'results':
            self.result_idx = (self.result_idx + 1) % len(self.echo_frames)
            self.im.set_data(self.echo_frames[self.result_idx])
            self.im2.set_data(self.tapir_frames[self.result_idx])
        # 'loading', 'selecting', 'tracking' — no frame update needed

        return []

    # ------------------------------------------------------------------
    # Button callbacks — main thread
    # ------------------------------------------------------------------

    def _on_pause_select(self, _):
        if self.state == 'tracking':
            return  # can't interrupt an ongoing run
        if not self._models_ready:
            self.status_txt.set_text('Models are still loading — please wait.')
            self.fig.canvas.draw_idle()
            return

        self.state = 'selecting'
        self.select_points = []
        self.play_idx = 0

        rgb0 = cv.cvtColor(self.frames_gray[0], cv.COLOR_GRAY2RGB)
        self._switch_to_single(rgb0, title='Click on the image to place tracking points')
        self.status_txt.set_text(
            'Left-click to add points.  Then click "Run Tracking".'
        )
        self.pts_txt.set_text('Points selected: 0')

    def _on_clear(self, _):
        if self.state != 'selecting':
            return
        self.select_points = []
        rgb0 = cv.cvtColor(self.frames_gray[0], cv.COLOR_GRAY2RGB)
        self._switch_to_single(rgb0, title='Click on the image to place tracking points')
        self.pts_txt.set_text('Points selected: 0')
        self.status_txt.set_text('Points cleared.  Click to select new points.')

    def _on_run_tracking(self, _):
        if self.state != 'selecting':
            return
        if not self.select_points:
            self.status_txt.set_text('Select at least one point before running tracking.')
            self.fig.canvas.draw_idle()
            return
        self.state = 'tracking'
        self.pts_txt.set_text('')
        self.status_txt.set_text(
            f'Running EchoTracker & TAPIR on {len(self.select_points)} point(s) — please wait…'
        )
        self.fig.canvas.draw_idle()
        threading.Thread(target=self._run_tracking_thread, daemon=True).start()

    # ------------------------------------------------------------------
    # Point selection click — main thread
    # ------------------------------------------------------------------

    def _on_click(self, event):
        if self.state != 'selecting':
            return
        if event.inaxes is not self.ax or event.button != 1:
            return
        x = int(np.clip(np.round(event.xdata), 0, self.display_w - 1))
        y = int(np.clip(np.round(event.ydata), 0, self.display_h - 1))
        self.select_points.append(np.array([x, y]))
        color = tuple(np.array(self.colormap[len(self.select_points) - 1]) / 255.0)
        self.ax.plot(x, y, 'o', color=color, markersize=9,
                     markeredgecolor='white', markeredgewidth=1.0)
        n = len(self.select_points)
        self.pts_txt.set_text(f'Points selected: {n}')
        self.status_txt.set_text(
            f'{n} point(s) placed.  Add more or click "Run Tracking".'
        )
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Inference (background thread)
    # ------------------------------------------------------------------

    def _run_tracking_thread(self):
        H, W = self.display_h, self.display_w

        # Frames tensor  (1, T, H, W, 1)
        frames_t = torch.from_numpy(self.frames_gray[..., np.newaxis]).unsqueeze(0)

        # Query points  (1, N, 2) normalised to [0, 1]
        pts = np.array(self.select_points, dtype=np.float32)
        qp = torch.from_numpy(pts).unsqueeze(0)
        qp[..., 0] /= W    # x → [0, 1]
        qp[..., 1] /= H    # y → [0, 1]

        # ---- EchoTracker ----
        trajs_echo = self.echotracker.infer(frames_t, qp, (256, 256))
        # trajs_echo: (1, N, T, 2); qp is restored to [0,1] by infer()
        trajs_echo_np = trajs_echo.squeeze(0).numpy()           # (N, T, 2)
        visibs_np = np.ones(trajs_echo_np.shape[:2], dtype=np.float32)  # (N, T)

        pd_echo = paint_vid(
            frames=self.frames_gray,
            points=trajs_echo_np,
            visibs=visibs_np,
            gray=True,
        )
        # pd_echo = add_text_to_frames(
        #     pd_echo, 'EchoTracker',
        #     color=(100, 255, 100), font_scale=0.8, position=(10, 25), thickness=2,
        # )

        # ---- TAPIR ----
        frames_rgb_t = frames_t.repeat(1, 1, 1, 1, 3)   # (1, T, H, W, 3)
        trajs_tapir = self.tapir.infer(frames_rgb_t, qp, (256, 256))
        trajs_tapir_np = trajs_tapir.squeeze(0).numpy()  # (N, T, 2)

        pd_tapir = paint_vid(
            frames=frames_rgb_t.squeeze(0).numpy(),
            points=trajs_tapir_np,
            visibs=visibs_np,
        )
        # pd_tapir = add_text_to_frames(
        #     pd_tapir, 'TAPIR',
        #     color=(100, 255, 100), font_scale=0.8, position=(10, 25), thickness=2,
        # )

        # Store separately so each model's frames keep their original aspect ratio
        self.echo_frames = pd_echo    # (T, H, W, 3)
        self.tapir_frames = pd_tapir  # (T, H, W, 3)

        # Save side-by-side to disk for reference
        out_path = 'results/output.mp4'
        media.write_video(out_path, np.concatenate([pd_echo, pd_tapir], axis=2), fps=20)
        print(f'Results saved to {out_path}')

        # Signal the animation tick to switch to results display
        self._tracking_done = True

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _switch_to_single(self, img, title=''):
        """Full-width single axes — used for playing and selecting."""
        # Hide second axes if present
        if self.ax2 is not None:
            self.ax2.set_visible(False)
        # Expand main axes to full width
        self.ax.set_position([0.01, 0.12, 0.98, 0.86])
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.axis('off')
        self.im = self.ax.imshow(img, aspect='equal')
        self.ax.set_title(title, color='white', pad=5)
        self.fig.canvas.draw_idle()

    def _switch_to_dual(self, echo_img, tapir_img):
        """Two side-by-side axes — used for results, each at native aspect ratio."""
        # Left axes: EchoTracker
        self.ax.set_position([0.01, 0.12, 0.475, 0.86])
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.axis('off')
        self.im = self.ax.imshow(echo_img, aspect='equal')
        self.ax.set_title('EchoTracker', color='#88ff88', pad=5, fontsize=11)

        # Right axes: TAPIR — create once, reuse after
        if self.ax2 is None:
            self.ax2 = self.fig.add_axes([0.515, 0.12, 0.475, 0.86])
        self.ax2.set_position([0.515, 0.12, 0.475, 0.86])
        self.ax2.set_visible(True)
        self.ax2.clear()
        self.ax2.set_facecolor('black')
        self.ax2.axis('off')
        self.im2 = self.ax2.imshow(tapir_img, aspect='equal')
        self.ax2.set_title('TAPIR', color='#88ff88', pad=5, fontsize=11)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python demo2.py <video_name>')
        sys.exit(1)

    video_path = f'data/{sys.argv[1]}.mp4'
    EchoTrackerDemo(video_path).show()
