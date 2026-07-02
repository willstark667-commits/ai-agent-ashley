# Ashley AI Studio — Implementation Analysis & Improvements
## Phase 1 Shell & GUI/UI Enhancement

**Date**: 2026-07-02  
**Status**: Analysis Complete → Ready for Implementation Phase 1 Finalization

---

## Current State Assessment

### What Exists
✅ **Core Infrastructure**
- `ashley_ai_studio.py` — Main orchestrator (HardwareMonitor, CheckpointManager, color utilities)
- `launcher.py` — Entry point with platform detection
- `core/` directory with agent, code generation, LLM bridges
- Architecture documentation (ARCHITECTURE.md)
- Requirements files with dependency specs
- Phase 1, 1-3, and completion tracking docs

✅ **Structural Decisions**
- Multi-mode system (GUI, Shell, Web)
- Hardware detection and optimization
- Checkpoint/savepoint system
- Platform-aware Windows UTF-8 handling

### What's Missing (Critical for Phase 1)
❌ **No actual GUI/shell implementation**
- `ashley_ai_studio.py` prints "Implementation in progress"
- No Tkinter GUI window created
- No pywebview shell running
- No real command-line interface
- No main loop or event handling

❌ **No image/video generator integration**
- No OpenAI API connectivity
- No image generation pipeline
- No video synthesis
- No real-time stream integration

❌ **No core AI agent functionality**
- Prompt memory database not implemented
- Multi-provider routing (OpenAI, Claude, Grok) is designed but not wired
- No conversation loop
- No code execution or result storage

---

## Phase 1 Build Plan: Shell + GUI + Starter Features

### Week 1-2: GUI/Shell Foundation

**Goal**: Get a working desktop app with menu navigation, settings, and status display.

#### **Option A: Tkinter (Python-Native, Simpler)**

```python
# phase1_shell/main_window.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from pathlib import Path

class AshleyStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Ashley AI Studio - Phase 1")
        self.root.geometry("1200x800")
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tabs
        self.tab_home = ttk.Frame(self.notebook)
        self.tab_projects = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)
        self.tab_console = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_home, text="Home")
        self.notebook.add(self.tab_projects, text="Projects")
        self.notebook.add(self.tab_settings, text="Settings")
        self.notebook.add(self.tab_console, text="Console")
        
        # Build each tab
        self._build_home_tab()
        self._build_projects_tab()
        self._build_settings_tab()
        self._build_console_tab()
    
    def _build_home_tab(self):
        # Status display, quick links, recent projects
        frame = ttk.LabelFrame(self.tab_home, text="Status", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Hardware info
        ttk.Label(frame, text="Hardware Info:", font=("Arial", 12, "bold")).pack()
        self.status_text = tk.Text(frame, height=10, width=80, state=tk.DISABLED)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Buttons
        btn_frame = ttk.Frame(self.tab_home)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Refresh Status", command=self.refresh_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="New Project", command=self.new_project).pack(side=tk.LEFT, padx=5)
    
    def _build_projects_tab(self):
        frame = ttk.LabelFrame(self.tab_projects, text="Projects", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview for project list
        columns = ("name", "type", "modified")
        self.project_tree = ttk.Treeview(frame, columns=columns, height=15)
        self.project_tree.column("#0", width=200)
        self.project_tree.heading("#0", text="Project Name")
        for col in columns:
            self.project_tree.column(col, width=150)
            self.project_tree.heading(col, text=col.capitalize())
        self.project_tree.pack(fill=tk.BOTH, expand=True)
    
    def _build_settings_tab(self):
        frame = ttk.LabelFrame(self.tab_settings, text="Configuration", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Theme
        ttk.Label(frame, text="Theme:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.theme_var = tk.StringVar(value="dark")
        ttk.Combobox(frame, textvariable=self.theme_var, values=["light", "dark"]).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # GPU Enable/Disable
        ttk.Label(frame, text="Use GPU:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, variable=self.gpu_var).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Save button
        ttk.Button(frame, text="Save Settings", command=self.save_settings).grid(row=10, column=0, columnspan=2, pady=20)
    
    def _build_console_tab(self):
        frame = ttk.LabelFrame(self.tab_console, text="Command Console", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Console output
        self.console_text = tk.Text(frame, height=20, width=100, bg="black", fg="lime")
        self.console_text.pack(fill=tk.BOTH, expand=True)
        
        # Command input
        input_frame = ttk.Frame(self.tab_console)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(input_frame, text="Command:").pack(side=tk.LEFT, padx=5)
        self.command_entry = ttk.Entry(input_frame)
        self.command_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_frame, text="Execute", command=self.execute_command).pack(side=tk.LEFT, padx=5)
    
    def refresh_status(self):
        from ashley_ai_studio import HardwareMonitor
        hm = HardwareMonitor()
        report = hm.get_report()
        
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert(tk.END, json.dumps(report, indent=2))
        self.status_text.config(state=tk.DISABLED)
    
    def new_project(self):
        messagebox.showinfo("New Project", "Project creation feature coming soon!")
    
    def save_settings(self):
        settings = {
            "theme": self.theme_var.get(),
            "use_gpu": self.gpu_var.get()
        }
        settings_file = Path("./config/settings.json")
        settings_file.parent.mkdir(exist_ok=True)
        settings_file.write_text(json.dumps(settings, indent=2))
        messagebox.showinfo("Settings", "Settings saved!")
    
    def execute_command(self):
        cmd = self.command_entry.get()
        self.console_text.insert(tk.END, f"\n$ {cmd}\n")
        # Add command execution here
        self.command_entry.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = AshleyStudio(root)
    root.mainloop()
```

---

#### **Option B: PyWebView (Web-Based, More Modern)**

```python
# phase1_shell/web_shell.py
import webview
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

class AshleyWebShell:
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        config_file = Path("./config/settings.json")
        if config_file.exists():
            return json.loads(config_file.read_text())
        return {"theme": "dark", "use_gpu": True}
    
    @app.route("/")
    def index():
        return render_template("index.html")
    
    @app.route("/api/status")
    def get_status():
        from ashley_ai_studio import HardwareMonitor
        hm = HardwareMonitor()
        return jsonify(hm.get_report())
    
    @app.route("/api/projects")
    def get_projects():
        projects_dir = Path("./projects")
        projects = []
        if projects_dir.exists():
            for proj_dir in projects_dir.iterdir():
                if proj_dir.is_dir():
                    projects.append({
                        "name": proj_dir.name,
                        "path": str(proj_dir),
                        "modified": proj_dir.stat().st_mtime
                    })
        return jsonify(projects)
    
    @app.route("/api/settings", methods=["GET", "POST"])
    def settings():
        if request.method == "GET":
            return jsonify(self.config)
        else:
            self.config = request.json
            config_file = Path("./config/settings.json")
            config_file.parent.mkdir(exist_ok=True)
            config_file.write_text(json.dumps(self.config, indent=2))
            return jsonify({"success": True})

if __name__ == "__main__":
    shell = AshleyWebShell()
    webview.create_window(
        "Ashley AI Studio - Phase 1",
        "http://localhost:5000",
        width=1200,
        height=800
    )
    webview.start()
    app.run(debug=True, port=5000)
```

```html
<!-- phase1_shell/templates/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Ashley AI Studio</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>Ashley AI Studio - Phase 1</h1>
        </header>
        
        <nav class="tabs">
            <button class="tab-btn active" onclick="switchTab('home')">Home</button>
            <button class="tab-btn" onclick="switchTab('projects')">Projects</button>
            <button class="tab-btn" onclick="switchTab('settings')">Settings</button>
            <button class="tab-btn" onclick="switchTab('console')">Console</button>
        </nav>
        
        <main>
            <!-- Home Tab -->
            <div id="tab-home" class="tab-content active">
                <h2>Status</h2>
                <pre id="status-display"></pre>
                <button onclick="refreshStatus()">Refresh</button>
            </div>
            
            <!-- Projects Tab -->
            <div id="tab-projects" class="tab-content">
                <h2>Projects</h2>
                <ul id="project-list"></ul>
                <button onclick="newProject()">New Project</button>
            </div>
            
            <!-- Settings Tab -->
            <div id="tab-settings" class="tab-content">
                <h2>Settings</h2>
                <label>
                    Theme:
                    <select id="theme-select">
                        <option>light</option>
                        <option>dark</option>
                    </select>
                </label>
                <label>
                    Use GPU:
                    <input type="checkbox" id="gpu-check">
                </label>
                <button onclick="saveSettings()">Save</button>
            </div>
            
            <!-- Console Tab -->
            <div id="tab-console" class="tab-content">
                <h2>Console</h2>
                <div id="console-output"></div>
                <input type="text" id="command-input" placeholder="Enter command...">
                <button onclick="executeCommand()">Execute</button>
            </div>
        </main>
    </div>
    
    <script src="/static/app.js"></script>
</body>
</html>
```

---

### Week 3-4: Image & Video Generator Integration

**Goal**: Connect to OpenAI API for image generation and basic video synthesis.

```python
# phase1_shell/image_generator.py
import asyncio
import openai
from pathlib import Path
from datetime import datetime
import json

class ImageGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key)
        self.history = []
        self.output_dir = Path("./generated_images")
        self.output_dir.mkdir(exist_ok=True)
    
    async def generate_image(self, prompt: str, size: str = "1024x1024", style: str = "photorealistic"):
        """Generate image using OpenAI DALL-E 3"""
        try:
            full_prompt = f"{style} style: {prompt}"
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=full_prompt,
                size=size,
                quality="hd",
                n=1
            )
            
            image_url = response.data[0].url
            image_data = {
                "prompt": prompt,
                "style": style,
                "size": size,
                "url": image_url,
                "timestamp": datetime.now().isoformat(),
                "revised_prompt": response.data[0].revised_prompt
            }
            
            # Save metadata
            self.history.append(image_data)
            self._save_history()
            
            # Download and save image
            self._download_image(image_url, image_data)
            
            return image_data
        
        except Exception as e:
            print(f"Image generation failed: {e}")
            return None
    
    async def generate_variations(self, image_path: str, n: int = 4):
        """Generate variations of an existing image"""
        try:
            with open(image_path, "rb") as img:
                response = self.client.images.create_variation(
                    image=img,
                    n=n,
                    size="1024x1024"
                )
            
            results = []
            for i, data in enumerate(response.data):
                self._download_image(data.url, {"index": i})
                results.append(data.url)
            
            return results
        except Exception as e:
            print(f"Variation generation failed: {e}")
            return []
    
    def _download_image(self, url: str, metadata: dict):
        """Download and save image locally"""
        try:
            import requests
            response = requests.get(url)
            filename = f"{metadata['timestamp'].replace(':', '-')}.png"
            filepath = self.output_dir / filename
            filepath.write_bytes(response.content)
            metadata["local_path"] = str(filepath)
        except Exception as e:
            print(f"Download failed: {e}")
    
    def _save_history(self):
        """Save generation history to JSON"""
        history_file = self.output_dir / "history.json"
        history_file.write_text(json.dumps(self.history, indent=2))
    
    def get_recent(self, n: int = 10) -> list:
        """Get recent generated images"""
        return self.history[-n:]
```

```python
# phase1_shell/video_generator.py
import subprocess
import json
from pathlib import Path
from datetime import datetime

class VideoGenerator:
    def __init__(self):
        self.output_dir = Path("./generated_videos")
        self.output_dir.mkdir(exist_ok=True)
        self.history = []
    
    def generate_from_images(self, image_paths: list, fps: int = 24, duration: int = 5):
        """Generate video from sequence of images using FFmpeg"""
        try:
            output_file = self.output_dir / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            # Create image list file for FFmpeg
            concat_file = self.output_dir / "concat.txt"
            concat_content = "\n".join([f"file '{img}'" for img in image_paths])
            concat_file.write_text(concat_content)
            
            # FFmpeg command
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-vf", f"fps={fps}",
                str(output_file)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            video_data = {
                "output_file": str(output_file),
                "image_count": len(image_paths),
                "fps": fps,
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            
            self.history.append(video_data)
            self._save_history()
            
            return video_data
        
        except Exception as e:
            print(f"Video generation failed: {e}")
            return None
    
    def add_audio_to_video(self, video_path: str, audio_path: str, output_path: str = None):
        """Add audio track to video using FFmpeg"""
        try:
            if not output_path:
                output_path = str(self.output_dir / f"video_with_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except Exception as e:
            print(f"Audio addition failed: {e}")
            return None
    
    def _save_history(self):
        history_file = self.output_dir / "history.json"
        history_file.write_text(json.dumps(self.history, indent=2))
```

---

## Key Improvements to Implement

### 1. **Real API Integration**
- Move from "design only" to actual OpenAI API calls
- Add multi-provider fallback (Claude, Gemini API)
- Implement rate limiting and cost tracking
- Add caching to reduce API calls

### 2. **Database + Persistence**
- SQLite backend for prompt memory
- Vector embeddings for semantic search (FAISS or ChromaDB)
- File index and metadata storage
- Training dataset organization

### 3. **Command Execution System**
- Safe subprocess execution for generated code
- Error capture and reporting
- Output logging and history

### 4. **Real-Time Streaming**
- WebRTC or WebSocket for live updates
- RTMP output for streaming platforms
- FFmpeg integration for video encoding

### 5. **Error Handling & Logging**
- Comprehensive exception hierarchy
- Structured logging with rotation
- Graceful degradation and fallbacks

---

## File Structure After Phase 1 Completion

```
ashley-ai-studio/
├── launcher.py                 # Main entry point
├── ashley_ai_studio.py         # Core orchestrator
├── requirements.txt            # Dependencies
├── config/
│   ├── settings.json          # User settings
│   └── config.yaml            # App configuration
├── core/
│   ├── ashley_agent.py        # AI agent core
│   ├── code_generator.py      # Code generation
│   ├── llm_bridge.py          # LLM routing
│   ├── odysseus_bridge.py     # Advanced orchestration
│   └── odysseus_launcher.py   # Alternative launcher
├── phase1_shell/
│   ├── shell.py               # Main GUI/Shell entry
│   ├── main_window.py         # Tkinter GUI (Option A)
│   ├── web_shell.py           # PyWebView (Option B)
│   ├── image_generator.py     # Image generation
│   ├── video_generator.py     # Video synthesis
│   ├── requirements.txt       # Phase 1 specific deps
│   ├── templates/
│   │   └── index.html         # Web UI
│   └── static/
│       ├── style.css          # Styling
│       └── app.js             # Frontend logic
├── generated_images/          # Output directory
├── generated_videos/          # Output directory
├── projects/                  # User projects
└── data/
    ├── ashley.db              # SQLite database
    └── vectors/               # Vector embeddings

```

---

## Next Immediate Actions

**Week 1:**
1. Choose GUI framework (Tkinter vs PyWebView)
2. Implement basic shell with menu navigation
3. Add hardware monitoring display
4. Create settings persistence

**Week 2:**
1. Integrate OpenAI Image API
2. Build image generation panel
3. Add image history browser
4. Implement caching system

**Week 3:**
1. Add video generation from image sequences
2. Integrate FFmpeg for video encoding
3. Add real-time progress tracking
4. Build output viewer

**Week 4:**
1. Refine UI/UX based on testing
2. Add error handling and logging
3. Optimize performance
4. Create user documentation

---

**Status**: Ready for Phase 1 Implementation  
**Owner**: willstark667-commits  
**Repository**: ai-agent-ashley

