#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Odysseus Launcher Module for Ashley AI Studio

Provides integration between Ashley AI Studio and the independent Odysseus program.
Allows launching, controlling, and communicating with Odysseus.

Usage:
    from core.odysseus_launcher import OdysseusLauncher
    
    launcher = OdysseusLauncher()
    launcher.launch()
    launcher.send_command("navigate", {"x": 10, "y": 20})
"""

import sys
import os
import json
import subprocess
import threading
import queue
import time
import platform
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import socket


class OdysseusLauncher:
    """
    Manages Odysseus program lifecycle and communication.
    
    Features:
    - Launch Odysseus as separate process
    - Send commands via IPC
    - Receive responses asynchronously
    - Monitor Odysseus status
    - Handle process lifecycle
    """
    
    def __init__(self, odysseus_dir: Optional[Path] = None):
        """
        Initialize Odysseus launcher.
        
        Args:
            odysseus_dir: Path to Odysseus directory. Defaults to ./odysseus
        """
        if odysseus_dir is None:
            odysseus_dir = Path(__file__).parent.parent / "odysseus"
        
        self.odysseus_dir = Path(odysseus_dir).resolve()
        self.odysseus_main = self.odysseus_dir / "odysseus_main.py"
        
        if not self.odysseus_dir.exists():
            raise FileNotFoundError(f"Odysseus directory not found: {self.odysseus_dir}")
        if not self.odysseus_main.exists():
            raise FileNotFoundError(f"Odysseus main script not found: {self.odysseus_main}")
        
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False
        self.mode = "ipc"  # Can be "ipc", "server", or "subprocess"
        self.server_port = 9999
        self.ipc_pipe: Optional[Any] = None
        self.response_queue = queue.Queue()
        self.listener_thread: Optional[threading.Thread] = None
        
        # Platform detection
        self.os_name = platform.system()
        self.is_win = self.os_name == "Windows"
    
    def launch(self, mode: str = "ipc", background: bool = True) -> bool:
        """
        Launch Odysseus program.
        
        Args:
            mode: Launch mode - "ipc" (default), "server", or "subprocess"
            background: Run in background if True
        
        Returns:
            True if launch successful, False otherwise
        """
        if self.is_running:
            print(f"[OdysseusLauncher] Already running on PID {self.process.pid}")
            return True
        
        self.mode = mode
        
        try:
            # Build command
            python_exe = sys.executable
            odysseus_script = str(self.odysseus_main)
            
            cmd = [python_exe, odysseus_script]
            
            if mode == "ipc":
                cmd.append("--ipc-mode")
            elif mode == "server":
                cmd.append("--server")
                cmd.extend(["--port", str(self.server_port)])
            
            # Launch process
            if background:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            else:
                self.process = subprocess.Popen(cmd)
            
            self.is_running = True
            
            # Start listener thread
            if mode == "ipc":
                self.listener_thread = threading.Thread(
                    target=self._listen_responses,
                    daemon=True
                )
                self.listener_thread.start()
            
            print(f"[OdysseusLauncher] Odysseus launched (PID: {self.process.pid}, Mode: {mode})")
            return True
            
        except Exception as e:
            print(f"[OdysseusLauncher] Launch failed: {e}")
            self.is_running = False
            return False
    
    def terminate(self) -> bool:
        """
        Terminate Odysseus process.
        
        Returns:
            True if termination successful, False otherwise
        """
        if not self.is_running or self.process is None:
            return False
        
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.is_running = False
            print(f"[OdysseusLauncher] Odysseus terminated")
            return True
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.is_running = False
            print(f"[OdysseusLauncher] Odysseus force-killed")
            return True
        except Exception as e:
            print(f"[OdysseusLauncher] Termination error: {e}")
            return False
    
    def send_command(
        self,
        command: str,
        args: Optional[Dict[str, Any]] = None,
        timeout: float = 5.0,
        async_mode: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Send command to Odysseus.
        
        Args:
            command: Command name
            args: Command arguments dict
            timeout: Response timeout in seconds
            async_mode: If True, return immediately without waiting for response
        
        Returns:
            Response dict or None if timeout/error
        """
        if not self.is_running or self.process is None:
            print("[OdysseusLauncher] Odysseus not running. Launch it first.")
            return None
        
        try:
            args = args or {}
            
            # Build command string
            cmd_str = f"{command}({', '.join(f'{k}={repr(v)}' for k, v in args.items())})\n"
            
            if self.mode == "ipc":
                # Send via stdin
                self.process.stdin.write(cmd_str)
                self.process.stdin.flush()
                
                if async_mode:
                    return {"status": "sent", "command": command}
                
                # Wait for response
                try:
                    response = self.response_queue.get(timeout=timeout)
                    return response
                except queue.Empty:
                    print(f"[OdysseusLauncher] Command timeout: {command}")
                    return None
            
            elif self.mode == "server":
                # Send via HTTP/socket
                return self._send_via_socket(command, args)
            
            else:
                return None
        
        except Exception as e:
            print(f"[OdysseusLauncher] Send command error: {e}")
            return None
    
    def _listen_responses(self):
        """Listen for responses from Odysseus subprocess."""
        if self.process is None or self.process.stdout is None:
            return
        
        try:
            for line in iter(self.process.stdout.readline, ""):
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Try to parse as JSON response
                try:
                    response = json.loads(line)
                    self.response_queue.put(response)
                except json.JSONDecodeError:
                    # Not JSON, might be debug output
                    pass
        
        except Exception as e:
            print(f"[OdysseusLauncher] Listener error: {e}")
        
        finally:
            self.is_running = False
    
    def _send_via_socket(self, command: str, args: Dict) -> Optional[Dict]:
        """Send command via socket (server mode)."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("localhost", self.server_port))
                
                cmd_dict = {"command": command, "args": args}
                cmd_json = json.dumps(cmd_dict)
                
                s.sendall(cmd_json.encode() + b"\n")
                
                response_data = s.recv(4096)
                response = json.loads(response_data.decode())
                
                return response
        except Exception as e:
            print(f"[OdysseusLauncher] Socket error: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get Odysseus status.
        
        Returns:
            Status dict
        """
        if not self.is_running:
            return {"running": False, "pid": None}
        
        return {
            "running": self.is_running,
            "pid": self.process.pid if self.process else None,
            "mode": self.mode,
        }
    
    def navigate_to(self, x: float, y: float, z: float = 0) -> Optional[Dict]:
        """
        Navigation shortcut: navigate to coordinates.
        
        Args:
            x, y, z: Target coordinates
        
        Returns:
            Response dict
        """
        return self.send_command("navigate", {"x": x, "y": y, "z": z})
    
    def scan_region(self, region_name: str) -> Optional[Dict]:
        """
        Exploration shortcut: scan region.
        
        Args:
            region_name: Name of region to scan
        
        Returns:
            Response dict
        """
        return self.send_command("scan_region", {"region_name": region_name})
    
    def add_waypoint(self, name: str, x: float, y: float, z: float = 0) -> Optional[Dict]:
        """
        Add navigation waypoint.
        
        Args:
            name: Waypoint name
            x, y, z: Waypoint coordinates
        
        Returns:
            Response dict
        """
        return self.send_command(
            "add_waypoint",
            {"name": name, "x": x, "y": y, "z": z}
        )
    
    def get_position(self) -> Optional[Dict]:
        """Get current Odysseus position."""
        return self.send_command("get_position")
    
    def list_waypoints(self) -> Optional[Dict]:
        """List all waypoints."""
        return self.send_command("list_waypoints")
    
    def list_discoveries(self) -> Optional[Dict]:
        """List all discovered regions."""
        return self.send_command("list_discoveries")


# Example usage
if __name__ == "__main__":
    # Demo
    launcher = OdysseusLauncher()
    
    # Launch Odysseus
    print("Launching Odysseus...")
    launcher.launch(mode="ipc", background=True)
    
    time.sleep(2)
    
    # Send some commands
    print("\nSending commands...")
    print(launcher.navigate_to(10, 20, 5))
    print(launcher.add_waypoint("Alpha", 15, 25, 10))
    print(launcher.scan_region("North"))
    print(launcher.get_position())
    
    time.sleep(1)
    
    # Cleanup
    print("\nTerminating Odysseus...")
    launcher.terminate()
