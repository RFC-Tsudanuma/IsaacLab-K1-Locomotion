# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data logger for recording robot state, foot trajectories, and command tracking."""

from __future__ import annotations

import torch
import numpy as np
from typing import  Dict, Any
import socket
import json
import threading
from queue import Queue, Full

class DataStreamSender:
    """Singleton class for sending JSON-serialized data to 127.0.0.1:9870 via UDP.

    This class uses a background thread to send data asynchronously, preventing
    blocking of the main simulation loop. Data is queued and sent in the order received.

    UDP is used for low-latency, connectionless transmission. No connection establishment
    is required, making it ideal for real-time data streaming.

    Usage:
        # Get the singleton instance and send data
        DataStreamSender.get_instance().send({"key": "value", "data": [1, 2, 3]})

        # Or use the convenience function
        send_data_stream({"robot_state": state_dict})
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, host: str = "127.0.0.1", port: int = 9870, queue_size: int = 100):
        """Initialize the data sender.

        Args:
            host: Target host address (default: "127.0.0.1")
            port: Target port number (default: 9870)
            queue_size: Maximum number of messages to queue (default: 100)
        """
        if DataStreamSender._instance is not None:
            raise RuntimeError("DataStreamSender is a singleton. Use get_instance() instead.")

        self.host = host
        self.port = port
        self.queue_size = queue_size

        # Message queue for async sending
        self._queue = Queue(maxsize=queue_size)

        # UDP socket (connectionless)
        self._socket = None
        self._enabled = True

        # Statistics
        self._send_count = 0
        self._error_count = 0
        self._last_error = None

        # Initialize UDP socket
        self._init_socket()

        # Start sender thread
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._running = True
        self._sender_thread.start()

        print(f"[DataStreamSender] Initialized (UDP). Target: {host}:{port}")

    @classmethod
    def get_instance(cls, host: str = "127.0.0.1", port: int = 9870, queue_size: int = 100):
        """Get the singleton instance.

        Args:
            host: Target host address (only used on first call)
            port: Target port number (only used on first call)
            queue_size: Maximum queue size (only used on first call)

        Returns:
            The singleton DataStreamSender instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(host, port, queue_size)
        return cls._instance

    def _init_socket(self):
        """Initialize UDP socket.

        UDP is connectionless, so no connection establishment is needed.
        """
        try:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass

            # Create UDP socket
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[DataStreamSender] UDP socket initialized for {self.host}:{self.port}")

        except Exception as e:
            self._last_error = str(e)
            print(f"[DataStreamSender] Failed to initialize socket: {e}")

    def _sender_loop(self):
        """Background thread loop for sending queued messages via UDP."""
        while self._running:
            try:
                # Get message from queue (blocking with timeout)
                try:
                    data = self._queue.get(timeout=0.1)
                except Exception:
                    continue

                if not self._enabled or self._socket is None:
                    continue

                # Send the data via UDP
                try:
                    # Convert numpy arrays to lists for JSON serialization
                    json_data = self._serialize_data(data)
                    message = json.dumps(json_data) + "\n"
                    message_bytes = message.encode('utf-8')

                    # UDP sendto - no connection needed
                    self._socket.sendto(message_bytes, (self.host, self.port))
                    self._send_count += 1

                except Exception as e:
                    # Log errors but continue
                    self._error_count += 1
                    self._last_error = str(e)
                    if self._error_count % 100 == 0:
                        print(f"[DataStreamSender] Send error: {e}")

            except Exception as e:
                print(f"[DataStreamSender] Unexpected error in sender loop: {e}")

    def _serialize_data(self, data: Any) -> Any:
        """Convert data to JSON-serializable format.

        Recursively converts numpy arrays and tensors to lists.

        Args:
            data: Data to serialize

        Returns:
            JSON-serializable version of the data
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy().tolist()
        elif isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_data(item) for item in data]
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        else:
            return data

    def send(self, data: Dict[str, Any]) -> bool:
        """Send data asynchronously to the target server.

        Args:
            data: Dictionary of data to send (will be JSON-serialized)

        Returns:
            True if data was queued successfully, False if queue is full
        """
        if not self._enabled:
            return False

        try:
            self._queue.put_nowait(data)
            return True
        except Full:
            # Queue is full, drop the message
            if self._error_count % 100 == 0:
                print("[DataStreamSender] Queue full, dropping message")
            return False

    def enable(self):
        """Enable data sending."""
        self._enabled = True
        print("[DataStreamSender] Enabled")

    def disable(self):
        """Disable data sending."""
        self._enabled = False
        print("[DataStreamSender] Disabled")

    def get_stats(self) -> Dict[str, Any]:
        """Get sender statistics.

        Returns:
            Dictionary containing statistics
        """
        return {
            'enabled': self._enabled,
            'send_count': self._send_count,
            'error_count': self._error_count,
            'queue_size': self._queue.qsize(),
            'last_error': self._last_error,
            'socket_initialized': self._socket is not None,
        }

    def close(self):
        """Close the sender and cleanup resources."""
        self._running = False
        self._enabled = False

        if self._sender_thread.is_alive():
            self._sender_thread.join(timeout=2.0)

        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass

        print("[DataStreamSender] Closed")


# Convenience function for easy access
def send_data_stream(data: Dict[str, Any]) -> bool:
    """Convenience function to send data using the singleton DataStreamSender.

    Args:
        data: Dictionary of data to send

    Returns:
        True if data was queued successfully, False otherwise

    Example:
        send_data_stream({
            "timestamp": time.time(),
            "robot_position": [x, y, z],
            "joint_angles": joint_array
        })
    """
    return DataStreamSender.get_instance().send(data)
