#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_srvs.srv import Trigger

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None
    import pandas as pd
else:
    pd = None

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _cv2_import_error = exc
else:
    _cv2_import_error = None


class JointStateBuffer:
    def __init__(self, maxlen=100):
        self.buf = deque(maxlen=maxlen)

    @staticmethod
    def _stamp_ns(msg):
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def push(self, msg: JointState):
        self.buf.append({
            'stamp_ns': self._stamp_ns(msg),
            'name': list(msg.name),
            'position': np.asarray(msg.position, dtype=np.float32).copy(),
        })

    def ready(self):
        return len(self.buf) > 0

    def latest(self):
        return self.buf[-1] if self.buf else None

    def nearest(self, target_ns):
        if not self.buf:
            return None
        return min(self.buf, key=lambda x: abs(x['stamp_ns'] - target_ns))

    def latest_before_or_nearest(self, target_ns):
        if not self.buf:
            return None
        prev = None
        for item in reversed(self.buf):
            if item['stamp_ns'] <= target_ns:
                prev = item
                break
        return prev if prev is not None else self.nearest(target_ns)

    def interpolate(self, target_ns):
        if not self.buf:
            return None
        if len(self.buf) == 1:
            return self.buf[-1]

        prev = None
        next_ = None
        for item in self.buf:
            if item['stamp_ns'] <= target_ns:
                prev = item
            if item['stamp_ns'] >= target_ns:
                next_ = item
                break

        if prev is None:
            return self.buf[0]
        if next_ is None:
            return self.buf[-1]
        if prev['stamp_ns'] == next_['stamp_ns']:
            return prev
        if prev['name'] != next_['name']:
            return self.nearest(target_ns)

        t0 = prev['stamp_ns']
        t1 = next_['stamp_ns']
        alpha = float(target_ns - t0) / float(t1 - t0)
        pos = (1.0 - alpha) * prev['position'] + alpha * next_['position']
        return {
            'stamp_ns': target_ns,
            'name': list(prev['name']),
            'position': pos.astype(np.float32),
        }


@dataclass
class EpisodeBuffers:
    image_stamps_ns: list
    actions: list
    states: list


class StreamingEpisodeVideoWriter:
    def __init__(self, path: Path, fps: int):
        if cv2 is None:
            raise RuntimeError(f'cv2 import failed: {_cv2_import_error}')
        self.path = Path(path)
        self.fps = int(fps)
        self.writer = None
        self.width = None
        self.height = None
        self.codec_name = None

    def _build_writer(self, width: int, height: int):
        codec_candidates = [
            ('avc1', 'h264'),
            ('mp4v', 'mp4v'),
            ('H264', 'h264'),
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for fourcc_name, codec_name in codec_candidates:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(str(self.path), fourcc, float(self.fps), (width, height))
            if writer is not None and writer.isOpened():
                self.writer = writer
                self.codec_name = codec_name
                self.width = width
                self.height = height
                return
            if writer is not None:
                writer.release()
        raise RuntimeError('failed to open cv2.VideoWriter for mp4 output')

    def write_rgb(self, rgb: np.ndarray):
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f'expected RGB image [H,W,3], got shape={rgb.shape}')
        height, width, _ = rgb.shape
        if self.writer is None:
            self._build_writer(width, height)
        if width != self.width or height != self.height:
            raise ValueError(
                f'image size changed during recording: got {(height, width)} expected {(self.height, self.width)}'
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class RunningFeatureStats:
    def __init__(self):
        self.data: dict[str, dict[str, np.ndarray | int]] = {}

    def update(self, name: str, values: np.ndarray):
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = arr.reshape(-1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            raise ValueError(f'unsupported stats ndim={arr.ndim} for {name}')

        arr64 = arr.astype(np.float64)
        cur_min = np.min(arr64, axis=0)
        cur_max = np.max(arr64, axis=0)
        cur_sum = np.sum(arr64, axis=0)
        cur_sumsq = np.sum(arr64 * arr64, axis=0)
        cur_count = int(arr64.shape[0])

        if name not in self.data:
            self.data[name] = {
                'min': cur_min,
                'max': cur_max,
                'sum': cur_sum,
                'sumsq': cur_sumsq,
                'count': cur_count,
            }
            return

        state = self.data[name]
        state['min'] = np.minimum(state['min'], cur_min)
        state['max'] = np.maximum(state['max'], cur_max)
        state['sum'] = state['sum'] + cur_sum
        state['sumsq'] = state['sumsq'] + cur_sumsq
        state['count'] = int(state['count']) + cur_count

    def to_jsonable(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, state in self.data.items():
            count = max(int(state['count']), 1)
            mean = state['sum'] / count
            var = np.maximum(state['sumsq'] / count - mean * mean, 0.0)
            std = np.sqrt(var)
            out[name] = {
                'min': _array_to_json(state['min']),
                'max': _array_to_json(state['max']),
                'mean': _array_to_json(mean),
                'std': _array_to_json(std),
                'count': count,
            }
        return out


def _array_to_json(arr: np.ndarray | float | int | bool):
    arr = np.asarray(arr)
    if arr.size == 1:
        value = arr.reshape(-1)[0]
        if arr.dtype.kind == 'b':
            return bool(value)
        if arr.dtype.kind in ('i', 'u'):
            return int(value)
        return float(value)
    if arr.dtype.kind == 'b':
        return [bool(x) for x in arr.tolist()]
    if arr.dtype.kind in ('i', 'u'):
        return [int(x) for x in arr.tolist()]
    return [float(x) for x in arr.tolist()]


def camera_info_to_dict(msg: CameraInfo):
    return {
        'width': int(msg.width),
        'height': int(msg.height),
        'distortion_model': msg.distortion_model,
        'd': list(msg.d),
        'k': list(msg.k),
        'r': list(msg.r),
        'p': list(msg.p),
    }


class LeRobotRecorderNode(Node):
    def __init__(self):
        super().__init__('lerobot_recorder')

        self.declare_parameter('repo_id', 'booster_vla_dataset')
        self.declare_parameter('root', '/home/master/Workspace/test_lxk')
        self.declare_parameter('task', 'pick up the object and place it into the target area')
        self.declare_parameter('fps', 30)
        self.declare_parameter('camera_key', 'head_rgb')
        self.declare_parameter('robot_type', 'booster_bimanual')
        self.declare_parameter('max_age_ms', 5.0)
        self.declare_parameter('lowdim_buffer_size', 100)
        self.declare_parameter('chunk_size', 1000)
        self.declare_parameter('episode_success_on_save', True)
        self.declare_parameter('codebase_version', 'v2.1')

        self.repo_id = self.get_parameter('repo_id').value
        self.root = self.get_parameter('root').value
        self.task = self.get_parameter('task').value
        self.fps = int(self.get_parameter('fps').value)
        self.camera_key = self.get_parameter('camera_key').value
        self.robot_type = self.get_parameter('robot_type').value
        self.max_age_ms = float(self.get_parameter('max_age_ms').value)
        self.lowdim_buffer_size = int(self.get_parameter('lowdim_buffer_size').value)
        self.chunk_size = int(self.get_parameter('chunk_size').value)
        self.episode_success_on_save = bool(self.get_parameter('episode_success_on_save').value)
        self.codebase_version = str(self.get_parameter('codebase_version').value)
        self.frame_period_ns = int(1e9 / self.fps)

        self.bridge = CvBridge()
        self.dataset_root = Path(self.root)
        self.meta_dir = self.dataset_root / 'meta'
        self.extra_camera_dir = self.dataset_root / 'extra' / 'camera'
        self.recording = False
        self.episode_idx = self._discover_next_episode_index()
        self.global_index = 0
        self.frame_idx = 0
        self.action_names = None
        self.state_names = None
        self.last_recorded_image_ns = None
        self.episode_start_ns = None
        self.video_height = None
        self.video_width = None
        self.video_codec = 'mp4v'

        self.color_msg = None
        self.color_info_msg = None
        self.arm_action = JointStateBuffer(self.lowdim_buffer_size)
        self.hand_action = JointStateBuffer(self.lowdim_buffer_size)
        self.arm_state = JointStateBuffer(self.lowdim_buffer_size)
        self.hand_state = JointStateBuffer(self.lowdim_buffer_size)

        self.current_episode = None
        self.current_video_writer = None
        self.current_video_temp_path = None
        self.tasks_map = self._load_tasks_map()
        self.global_stats = RunningFeatureStats()
        self.episode_stats_cache = self._load_episode_stats_cache()

        self.recorded_frames = 0
        self.dropped_not_ready = 0
        self.dropped_fps = 0
        self.dropped_sync = 0
        self.sync_warn_counts = {
            'arm_action': 0,
            'hand_action': 0,
            'arm_state': 0,
            'hand_state': 0,
        }

        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_cb, 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.color_info_cb, 10)
        self.create_subscription(JointState, '/teleop/arm_action', self.arm_action_cb, 10)
        self.create_subscription(JointState, '/teleop/hand_action', self.hand_action_cb, 10)
        self.create_subscription(JointState, '/robot/arm_state', self.arm_state_cb, 10)
        self.create_subscription(JointState, '/robot/hand_state', self.hand_state_cb, 10)

        self.create_service(Trigger, '/start_episode', self.start_episode_cb)
        self.create_service(Trigger, '/save_episode', self.save_episode_cb)
        self.create_service(Trigger, '/discard_episode', self.discard_episode_cb)
        self.create_service(Trigger, '/finalize_dataset', self.finalize_dataset_cb)

        self.create_timer(5.0, self.log_stats)
        self.get_logger().info(
            f'lerobot_recorder_v2 started; fps={self.fps}, max_age_ms={self.max_age_ms}, '
            f'lowdim_buffer_size={self.lowdim_buffer_size}, next_episode={self.episode_idx}'
        )

    @staticmethod
    def stamp_ns(msg):
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _discover_next_episode_index(self) -> int:
        data_root = Path(self.root) / 'data'
        if not data_root.exists():
            return 0
        max_idx = -1
        for pq_path in data_root.glob('chunk-*/episode_*.parquet'):
            stem = pq_path.stem
            if stem.startswith('episode_'):
                try:
                    max_idx = max(max_idx, int(stem.split('_')[1]))
                except Exception:
                    pass
        return max_idx + 1

    def _load_tasks_map(self) -> dict[str, int]:
        tasks_path = self.dataset_root / 'meta' / 'tasks.jsonl'
        tasks = {}
        if tasks_path.exists():
            for line in tasks_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    tasks[str(row['task'])] = int(row['task_index'])
                except Exception:
                    continue
        return tasks

    def _load_episode_stats_cache(self) -> list[dict[str, Any]]:
        stats_path = self.dataset_root / 'meta' / 'episodes_stats.jsonl'
        rows = []
        if stats_path.exists():
            for line in stats_path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

    def log_stats(self):
        if not self.recording:
            return
        self.get_logger().info(
            'record stats: '
            f'recorded={self.recorded_frames}, '
            f'dropped_not_ready={self.dropped_not_ready}, '
            f'dropped_fps={self.dropped_fps}, '
            f'dropped_sync={self.dropped_sync}, '
            f'sync_warn_counts={self.sync_warn_counts}'
        )

    def color_cb(self, msg):
        self.color_msg = msg
        if self.recording:
            self.try_record_with_image(msg)

    def color_info_cb(self, msg):
        self.color_info_msg = msg

    def arm_action_cb(self, msg):
        self.arm_action.push(msg)

    def hand_action_cb(self, msg):
        self.hand_action.push(msg)

    def arm_state_cb(self, msg):
        self.arm_state.push(msg)

    def hand_state_cb(self, msg):
        self.hand_state.push(msg)

    def ready(self):
        return all([
            self.color_msg is not None,
            self.arm_action.ready(),
            self.hand_action.ready(),
            self.arm_state.ready(),
            self.hand_state.ready(),
        ])

    def merged_action_names(self):
        return list(self.arm_action.latest()['name']) + list(self.hand_action.latest()['name'])

    def merged_state_names(self):
        return list(self.arm_state.latest()['name']) + list(self.hand_state.latest()['name'])

    def ensure_dataset_layout(self):
        if not self.ready():
            raise RuntimeError('Waiting for color image, arm/hand action, and arm/hand state topics')

        rgb = self.bridge.imgmsg_to_cv2(self.color_msg, desired_encoding='rgb8')
        h, w, c = rgb.shape
        self.video_height = int(h)
        self.video_width = int(w)

        if self.action_names is None:
            self.action_names = self.merged_action_names()
        if self.state_names is None:
            self.state_names = self.merged_state_names()

        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.extra_camera_dir.mkdir(parents=True, exist_ok=True)
        self.write_calibration_once()
        self.write_tasks_jsonl()
        self.write_info_json()
        self.write_stats_json()
        self.write_episodes_jsonl()
        self.write_episodes_stats_jsonl()

    def write_calibration_once(self):
        if self.color_info_msg is not None:
            with open(self.extra_camera_dir / 'color_camera_info.json', 'w', encoding='utf-8') as f:
                json.dump(camera_info_to_dict(self.color_info_msg), f, indent=2)

        with open(self.extra_camera_dir / 'record_spec.json', 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'task': self.task,
                    'fps': self.fps,
                    'action_names': self.action_names,
                    'state_names': self.state_names,
                    'camera_key': self.camera_key,
                    'max_age_ms': self.max_age_ms,
                    'lowdim_buffer_size': self.lowdim_buffer_size,
                    'format': self.codebase_version,
                },
                f,
                indent=2,
            )

    def get_or_create_task_index(self, task: str) -> int:
        if task in self.tasks_map:
            return self.tasks_map[task]
        task_index = len(self.tasks_map)
        self.tasks_map[task] = task_index
        self.write_tasks_jsonl()
        return task_index

    def write_tasks_jsonl(self):
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        path = self.meta_dir / 'tasks.jsonl'
        rows = [
            {'task_index': int(task_index), 'task': task}
            for task, task_index in sorted(self.tasks_map.items(), key=lambda kv: kv[1])
        ]
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def write_episodes_jsonl(self):
        path = self.meta_dir / 'episodes.jsonl'
        rows = []
        existing = {}
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    existing[int(row['episode_index'])] = row
                except Exception:
                    continue
        rows = [existing[k] for k in sorted(existing.keys())]
        with open(path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def append_episode_jsonl(self, episode_index: int, task: str, length: int):
        path = self.meta_dir / 'episodes.jsonl'
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'episode_index': int(episode_index),
                'tasks': [task],
                'length': int(length),
            }, ensure_ascii=False) + '\n')

    def write_episodes_stats_jsonl(self):
        path = self.meta_dir / 'episodes_stats.jsonl'
        with open(path, 'w', encoding='utf-8') as f:
            for row in self.episode_stats_cache:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def append_episode_stats_jsonl(self, row: dict[str, Any]):
        self.episode_stats_cache.append(row)
        path = self.meta_dir / 'episodes_stats.jsonl'
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def write_info_json(self):
        total_episodes = self._count_episode_files()
        total_frames = self.global_index
        total_chunks = max(1, math.ceil(max(total_episodes, 1) / max(self.chunk_size, 1)))
        total_videos = total_episodes

        features = {
            f'observation.images.{self.camera_key}': {
                'dtype': 'video',
                'shape': [self.video_height, self.video_width, 3],
                'names': ['height', 'width', 'channel'],
                'video_info': {
                    'video.fps': float(self.fps),
                    'video.codec': self.video_codec,
                    'video.pix_fmt': 'yuv420p',
                    'video.is_depth_map': False,
                    'has_audio': False,
                },
            },
            'observation.state': {
                'dtype': 'float32',
                'shape': [len(self.state_names)],
                'names': self.state_names,
                'fps': float(self.fps),
            },
            'action': {
                'dtype': 'float32',
                'shape': [len(self.action_names)],
                'names': self.action_names,
                'fps': float(self.fps),
            },
            'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None, 'fps': float(self.fps)},
            'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None, 'fps': float(self.fps)},
            'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None, 'fps': float(self.fps)},
            'next.reward': {'dtype': 'float32', 'shape': [1], 'names': None, 'fps': float(self.fps)},
            'next.done': {'dtype': 'bool', 'shape': [1], 'names': None, 'fps': float(self.fps)},
            'next.success': {'dtype': 'bool', 'shape': [1], 'names': None, 'fps': float(self.fps)},
            'index': {'dtype': 'int64', 'shape': [1], 'names': None, 'fps': float(self.fps)},
            'task_index': {'dtype': 'int64', 'shape': [1], 'names': None, 'fps': float(self.fps)},
        }

        info = {
            'codebase_version': self.codebase_version,
            'robot_type': self.robot_type,
            'total_episodes': int(total_episodes),
            'total_frames': int(total_frames),
            'total_tasks': int(len(self.tasks_map)),
            'total_videos': int(total_videos),
            'total_chunks': int(total_chunks),
            'chunks_size': int(self.chunk_size),
            'fps': int(self.fps),
            'splits': {'train': f'0:{int(total_episodes)}'},
            'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
            'video_path': f'videos/chunk-{{episode_chunk:03d}}/observation.images.{self.camera_key}/episode_{{episode_index:06d}}.mp4',
            'features': features,
            'data_files_size_in_mb': self._dir_size_mb(self.dataset_root / 'data'),
            'video_files_size_in_mb': self._dir_size_mb(self.dataset_root / 'videos'),
        }
        with open(self.meta_dir / 'info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def write_stats_json(self):
        with open(self.meta_dir / 'stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.global_stats.to_jsonable(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def _dir_size_mb(path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for fp in path.rglob('*'):
            if fp.is_file():
                total += fp.stat().st_size
        return int(math.ceil(total / (1024 * 1024))) if total > 0 else 0

    def _count_episode_files(self) -> int:
        data_root = self.dataset_root / 'data'
        if not data_root.exists():
            return 0
        return sum(1 for _ in data_root.glob('chunk-*/episode_*.parquet'))

    def _episode_chunk(self, episode_index: int) -> int:
        return int(episode_index // self.chunk_size)

    def _data_path_for_episode(self, episode_index: int) -> Path:
        chunk = self._episode_chunk(episode_index)
        return self.dataset_root / 'data' / f'chunk-{chunk:03d}' / f'episode_{episode_index:06d}.parquet'

    def _video_path_for_episode(self, episode_index: int) -> Path:
        chunk = self._episode_chunk(episode_index)
        return (
            self.dataset_root
            / 'videos'
            / f'chunk-{chunk:03d}'
            / f'observation.images.{self.camera_key}'
            / f'episode_{episode_index:06d}.mp4'
        )

    def start_episode_cb(self, request, response):
        try:
            self.ensure_dataset_layout()
            if self.recording:
                raise RuntimeError('episode already recording')

            temp_dir = self.dataset_root / '.tmp_videos'
            temp_dir.mkdir(parents=True, exist_ok=True)
            tmp_file = tempfile.NamedTemporaryFile(
                prefix=f'episode_{self.episode_idx:06d}_', suffix='.mp4', dir=temp_dir, delete=False
            )
            tmp_path = Path(tmp_file.name)
            tmp_file.close()

            self.current_video_temp_path = tmp_path
            self.current_video_writer = StreamingEpisodeVideoWriter(tmp_path, self.fps)
            self.current_episode = EpisodeBuffers(image_stamps_ns=[], actions=[], states=[])
            self.frame_idx = 0
            self.recorded_frames = 0
            self.dropped_not_ready = 0
            self.dropped_fps = 0
            self.dropped_sync = 0
            self.sync_warn_counts = {k: 0 for k in self.sync_warn_counts}
            self.last_recorded_image_ns = None
            self.episode_start_ns = None
            self.recording = True
            response.success = True
            response.message = f'episode {self.episode_idx} started'
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f'start_episode failed: {exc}')
        return response

    def save_episode_cb(self, request, response):
        try:
            self.recording = False
            if self.current_episode is None or self.frame_idx == 0:
                response.success = False
                response.message = 'no frames recorded'
                return response

            self.current_video_writer.close()
            self.current_video_writer = None

            task_index = self.get_or_create_task_index(self.task)
            episode_index = int(self.episode_idx)
            data_path = self._data_path_for_episode(episode_index)
            video_path = self._video_path_for_episode(episode_index)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.current_video_temp_path), str(video_path))
            self.current_video_temp_path = None

            arrays = self._build_episode_arrays(episode_index, task_index)
            self._write_parquet(data_path, arrays)
            self._update_stats(arrays)
            self.append_episode_jsonl(episode_index, self.task, self.frame_idx)
            self.append_episode_stats_jsonl(self._build_episode_stats_row(episode_index, arrays))
            self.write_stats_json()
            self.write_info_json()

            response.success = True
            response.message = f'episode {episode_index} saved with {self.frame_idx} frames (v2.1 format)'
            self.get_logger().info(response.message)

            self.episode_idx += 1
            self.frame_idx = 0
            self.current_episode = None
            self.last_recorded_image_ns = None
            self.episode_start_ns = None
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f'save_episode failed: {exc}')
            self._cleanup_partial_episode_files()
        return response

    def finalize_dataset_cb(self, request, response):
        try:
            self.ensure_dataset_layout()
            self.write_tasks_jsonl()
            self.write_episodes_stats_jsonl()
            self.write_stats_json()
            self.write_info_json()
            response.success = True
            response.message = 'dataset metadata refreshed'
            self.get_logger().info(response.message)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f'finalize_dataset failed: {exc}')
        return response

    def discard_episode_cb(self, request, response):
        self.recording = False
        self.frame_idx = 0
        self.last_recorded_image_ns = None
        self.episode_start_ns = None
        self.current_episode = None
        if self.current_video_writer is not None:
            self.current_video_writer.close()
            self.current_video_writer = None
        if self.current_video_temp_path is not None and self.current_video_temp_path.exists():
            self.current_video_temp_path.unlink(missing_ok=True)
            self.current_video_temp_path = None
        response.success = True
        response.message = 'episode discarded'
        self.get_logger().info(response.message)
        return response

    def _cleanup_partial_episode_files(self):
        if self.current_video_writer is not None:
            self.current_video_writer.close()
            self.current_video_writer = None
        if self.current_video_temp_path is not None and self.current_video_temp_path.exists():
            self.current_video_temp_path.unlink(missing_ok=True)
        self.current_video_temp_path = None
        self.current_episode = None

    def get_synced_lowdim(self, image_ns):
        picks = {
            'arm_action': self.arm_action.latest_before_or_nearest(image_ns),
            'hand_action': self.hand_action.latest_before_or_nearest(image_ns),
            'arm_state': self.arm_state.interpolate(image_ns),
            'hand_state': self.hand_state.interpolate(image_ns),
        }

        for name, item in picks.items():
            if item is None:
                return None
            skew_ms = abs(image_ns - item['stamp_ns']) / 1_000_000.0
            if skew_ms > self.max_age_ms:
                self.sync_warn_counts[name] += 1
                return None

        action_names = list(picks['arm_action']['name']) + list(picks['hand_action']['name'])
        state_names = list(picks['arm_state']['name']) + list(picks['hand_state']['name'])

        action = np.concatenate([
            picks['arm_action']['position'],
            picks['hand_action']['position'],
        ], axis=0).astype(np.float32)

        state = np.concatenate([
            picks['arm_state']['position'],
            picks['hand_state']['position'],
        ], axis=0).astype(np.float32)

        return {
            'action_names': action_names,
            'state_names': state_names,
            'action': action,
            'state': state,
        }

    def try_record_with_image(self, image_msg):
        if not self.ready() or self.current_episode is None or self.current_video_writer is None:
            self.dropped_not_ready += 1
            return

        image_ns = self.stamp_ns(image_msg)

        if self.last_recorded_image_ns is not None:
            if image_ns - self.last_recorded_image_ns < self.frame_period_ns:
                self.dropped_fps += 1
                return

        synced = self.get_synced_lowdim(image_ns)
        if synced is None:
            self.dropped_sync += 1
            return

        if synced['action_names'] != self.action_names or synced['state_names'] != self.state_names:
            self.get_logger().error('action/state JointState names changed during recording')
            return

        rgb = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        self.current_video_writer.write_rgb(rgb)
        self.video_codec = self.current_video_writer.codec_name or self.video_codec

        if self.episode_start_ns is None:
            self.episode_start_ns = int(image_ns)

        self.current_episode.image_stamps_ns.append(int(image_ns))
        self.current_episode.actions.append(synced['action'].astype(np.float32).copy())
        self.current_episode.states.append(synced['state'].astype(np.float32).copy())

        self.last_recorded_image_ns = image_ns
        self.frame_idx += 1
        self.recorded_frames += 1

    def _build_episode_arrays(self, episode_index: int, task_index: int) -> dict[str, np.ndarray]:
        if self.current_episode is None:
            raise RuntimeError('no current episode')

        length = len(self.current_episode.image_stamps_ns)
        if length == 0:
            raise RuntimeError('episode has zero frames')

        image_stamps = np.asarray(self.current_episode.image_stamps_ns, dtype=np.int64)
        actions = np.stack(self.current_episode.actions, axis=0).astype(np.float32)
        states = np.stack(self.current_episode.states, axis=0).astype(np.float32)
        timestamps = ((image_stamps - image_stamps[0]).astype(np.float64) / 1e9).astype(np.float32)
        frame_index = np.arange(length, dtype=np.int64)
        episode_index_arr = np.full(length, int(episode_index), dtype=np.int64)
        task_index_arr = np.full(length, int(task_index), dtype=np.int64)
        global_index = np.arange(self.global_index, self.global_index + length, dtype=np.int64)
        next_reward = np.zeros(length, dtype=np.float32)
        next_done = np.zeros(length, dtype=np.bool_)
        next_success = np.zeros(length, dtype=np.bool_)
        next_done[-1] = True
        next_success[-1] = bool(self.episode_success_on_save)
        self.global_index += length

        return {
            'action': actions,
            'observation.state': states,
            'episode_index': episode_index_arr,
            'frame_index': frame_index,
            'timestamp': timestamps,
            'next.reward': next_reward,
            'next.done': next_done,
            'next.success': next_success,
            'index': global_index,
            'task_index': task_index_arr,
        }

    def _write_parquet(self, path: Path, arrays: dict[str, np.ndarray]):
        path.parent.mkdir(parents=True, exist_ok=True)

        if pa is not None and pq is not None:
            arrow_arrays = []
            arrow_fields = []

            for name, arr in arrays.items():
                np_arr = np.asarray(arr)

                if name in ('action', 'observation.state'):
                    np_arr = np.asarray(np_arr, dtype=np.float32)
                    rows = [np.asarray(row, dtype=np.float32).tolist() for row in np_arr]
                    field_type = pa.list_(pa.float32())
                    arrow_arr = pa.array(rows, type=field_type)
                elif name in ('timestamp', 'next.reward'):
                    np_arr = np.asarray(np_arr, dtype=np.float32)
                    field_type = pa.float32()
                    arrow_arr = pa.array(np_arr, type=field_type)
                elif name in ('episode_index', 'frame_index', 'index', 'task_index'):
                    np_arr = np.asarray(np_arr, dtype=np.int64)
                    field_type = pa.int64()
                    arrow_arr = pa.array(np_arr, type=field_type)
                elif name in ('next.done', 'next.success'):
                    np_arr = np.asarray(np_arr, dtype=np.bool_)
                    field_type = pa.bool_()
                    arrow_arr = pa.array(np_arr, type=field_type)
                else:
                    if np_arr.ndim == 2 and np.issubdtype(np_arr.dtype, np.floating):
                        np_arr = np.asarray(np_arr, dtype=np.float32)
                        rows = [np.asarray(row, dtype=np.float32).tolist() for row in np_arr]
                        field_type = pa.list_(pa.float32())
                        arrow_arr = pa.array(rows, type=field_type)
                    elif np.issubdtype(np_arr.dtype, np.floating):
                        np_arr = np.asarray(np_arr, dtype=np.float32)
                        field_type = pa.float32()
                        arrow_arr = pa.array(np_arr, type=field_type)
                    elif np.issubdtype(np_arr.dtype, np.integer):
                        np_arr = np.asarray(np_arr, dtype=np.int64)
                        field_type = pa.int64()
                        arrow_arr = pa.array(np_arr, type=field_type)
                    elif np.issubdtype(np_arr.dtype, np.bool_):
                        np_arr = np.asarray(np_arr, dtype=np.bool_)
                        field_type = pa.bool_()
                        arrow_arr = pa.array(np_arr, type=field_type)
                    else:
                        field_type = None
                        arrow_arr = pa.array(np_arr.tolist())

                arrow_fields.append(pa.field(name, field_type or arrow_arr.type))
                arrow_arrays.append(arrow_arr)

            table = pa.Table.from_arrays(arrow_arrays, schema=pa.schema(arrow_fields))
            pq.write_table(table, path)
            return

        if pd is None:
            raise RuntimeError('neither pyarrow nor pandas is available for parquet writing')

        data = {}
        for name, arr in arrays.items():
            np_arr = np.asarray(arr)
            if np.issubdtype(np_arr.dtype, np.floating):
                np_arr = np.asarray(np_arr, dtype=np.float32)
            elif np.issubdtype(np_arr.dtype, np.integer):
                np_arr = np.asarray(np_arr, dtype=np.int64)
            elif np.issubdtype(np_arr.dtype, np.bool_):
                np_arr = np.asarray(np_arr, dtype=np.bool_)

            if np_arr.ndim == 2:
                data[name] = [row.tolist() for row in np_arr]
            else:
                data[name] = np_arr.tolist()

        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)

    def _update_stats(self, arrays: dict[str, np.ndarray]):
        for name, arr in arrays.items():
            self.global_stats.update(name, np.asarray(arr))

    def _build_episode_stats_row(self, episode_index: int, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        episode_stats = RunningFeatureStats()
        for name, arr in arrays.items():
            episode_stats.update(name, np.asarray(arr))
        return {
            'episode_index': int(episode_index),
            'stats': episode_stats.to_jsonable(),
        }

    def destroy_node(self):
        self.get_logger().info('shutting down recorder')
        if self.current_video_writer is not None:
            self.current_video_writer.close()
        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)
    node = LeRobotRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
