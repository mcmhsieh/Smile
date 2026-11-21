"""
Pipeline server used by stages of the processing pipeline for
https://github.com/mcmhsieh/Smile to display their working states and provide
information for each stage to wait for their predecessor to complete.

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import sys
import os
import argparse
import pathlib
import platform
import subprocess
import threading
import queue
import functools
import socket
import http.server
import time
import collections
import json
import requests
import tkinter as tk
from tkinter import ttk


PIPELINE_SERVER_FILEPATH = __file__
# Note that localhost is slow to resolve
SERVER_NAME = '127.0.0.1'
SERVER_PORT = 8081

SYS_EXECUTABLE = pathlib.Path(sys.executable)
# os.name returns which implementation of the os APIs was imported
# sys.platform returns build time information
# The platform module returns runtime diagnostic information intended for humans to read
# and is currently not dependable for machine interpretation because return values may change over time.
SYS_EXECUTABLE_NO_CONSOLE = SYS_EXECUTABLE.parent / 'pythonw.exe'
if os.name == 'nt' and SYS_EXECUTABLE.name.lower() == 'python.exe' and SYS_EXECUTABLE_NO_CONSOLE.exists():
    SYS_EXECUTABLE = SYS_EXECUTABLE_NO_CONSOLE

class PipelineServer(http.server.HTTPServer):
    def __init__(self, *args, window=None, **kwargs):
        self.pipeline_queue = {}
        self.window = window
        super().__init__(*args, **kwargs)

    def server_bind(self, *args, **kwargs):
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            # Unlike Linux, on Windows the SO_REUSEADDR option forcibly binds sockets
            # when they are in any state, not just TIME_WAIT.
            # Apply SO_EXCLUSIVEADDRUSE instead of SO_REUSEADDR.
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            # Set SO_REUSEADDR option to bind to sockets in TIME_WAIT state
            self.allow_reuse_address = True
        super().server_bind(*args, **kwargs)

class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        response = json.dumps(self.server.pipeline_queue).encode('utf8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response)

    def do_POST(self):
        workspace_stage, state = json.loads(self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8'))
        if state is not None:
            self.server.pipeline_queue[workspace_stage] = state
        else:
            del self.server.pipeline_queue[workspace_stage]
        self.server.window.schedule_update(self.server.pipeline_queue)
        self.send_response(200)
        self.end_headers()

def httpd_main(httpd):
    try:
        httpd.serve_forever()
    finally:
        window.schedule_destroy()

def start_pipeline_server():
    # Create an independent / daemon process by starting a child process and exiting its parent process.
    # For some reason, creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    # still creates a child process within the process tree.
    subprocess.Popen([SYS_EXECUTABLE, PIPELINE_SERVER_FILEPATH])
    start_time = time.time()
    while True:
        try:
            get_queue_from_pipeline_server()
            break
        except requests.exceptions.ConnectionError:
            assert time.time() < start_time + 30, 'timeout waiting for pipeline server to respond'
            print('waiting for pipeline server to respond')
            time.sleep(1)

def post_to_pipeline_server(obj):
    requests.post(f'http://{SERVER_NAME}:{SERVER_PORT}', data=json.dumps(obj), timeout=10)

def get_queue_from_pipeline_server():
    r = requests.get(f'http://{SERVER_NAME}:{SERVER_PORT}', timeout=10)
    return r.json()

class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.update_queue = queue.Queue()
        self.terminate_on_empty_timeout = 30
        self.terminating = False
        self.last_empty_time = None

        self.title('Pipeline Server')
        self.geometry('500x300+100+100')

        icon_data = """iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAALHRFWHRDcmVhdGlvbiBUaW1lAFNhdCAyMSBOb3YgMjAyNiAxMjowOToyNCAtMDAwMEXK5bsAAAAHdElNRQfqCxUOCDiBWRtkAAAACXBIWXMAAAsSAAALEgHS3X78AAAABGdBTUEAALGPC/xhBQAAAGxQTFRF////9/f37+/31tbWSkpKMTExe3t77+/vUlJS3t7e5+fnKSkpAAAAc3NzlJSUQkJCxsbGCAgIvb29a2trzs7OhISEISEhjIyMOTk5GBgYpaWl3t7n1tbenJycY2NjY2NrtbW1Wlpaa2tzra2teVnMigAAAAF0Uk5TAEDm2GYAAAFcSURBVHjajZPrloMgDIRrFKk6VFGk1Hbtun3/d1yUi1rPXvJDcuQDJgM5nWwkP8TJRfJL7OaJ6C1xhP/5mlhlhCJSwlRselEA3JgayGw0wFcHmDGTMA9PLN9CIy8pIVUDtbJJmUMXK0AZBrclnyBsxvVlwJkioPAMsngNZVf0yMc58UCPNMpW+Fg42aKPwKjXuqjg89DioscITGwFwlGyZ1MEnvoA0Gequ1WD5EeX7R4RGJzgfSgMEWhcvo9hUyaX4ggId65zktVHlTVbnUzGo0oux81lfaB5B5rF0QA8nK3b6LG5zYQq40TQ/e4TU9EGSITnB/iKC/jCPNDi6tbdbm6vK9odQHopSkGIxR9i4XrCo87nFdRBKXQ07xhUB6CQ1qtUZkSZTK1LsngDSOBin1szG1DzC6L3sW/KCmx5sBZlqMoIRCLNWL4YznN2Ltb5besds+Q/zftH+38DM9wSamwR8ikAAAAASUVORK5CYII="""
        self.iconphoto(True, tk.PhotoImage(data=icon_data))

        self.tree = ttk.Treeview(self, columns=('workspace_stage', 'state'), show='headings')

        self.tree.heading('workspace_stage', text='Workspace / Stage')
        self.tree.heading('state', text='State')

        self.tree.column('workspace_stage', width=350, anchor=tk.CENTER)
        self.tree.column('state', width=100, anchor=tk.CENTER)

        self.tree.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.style.configure('Treeview.Heading', background='light blue', relief='sunken')

    def schedule_destroy(self):
        if self.terminating:
            return

        self.terminating = True
        self.after(0, self.destroy)

    def schedule_update(self, pipeline_queue):
        if self.terminating:
            return

        self.update_queue.put(dict(pipeline_queue))
        self.after(0, self.update)

    def update(self):
        if self.terminating:
            return

        pipeline_queue = self.update_queue.get()

        if len(pipeline_queue) > 0:
            self.last_empty_time = None
        elif self.last_empty_time is None:
            self.last_empty_time = time.time()
            self.after(self.terminate_on_empty_timeout * 1000, self.check_terminate_timeout)

        states = collections.Counter(state for _, state in pipeline_queue.items())
        if states['running'] > 0:
            self.title(f'⏩︎{states["running"]} Pipeline Server')
        elif states['waiting'] > 0:
            self.title(f'⏯{states["waiting"]} Pipeline Server')
        else:
            self.title('Pipeline Server')

        self.tree.delete(*self.tree.get_children())

        for row_idx, (workspace_stage, state) in enumerate(pipeline_queue.items()):
            row_tag = ['gray', 'white'][row_idx % 2]
            self.tree.insert('', tk.END, values=(workspace_stage, state), tag=row_tag)
        self.tree.tag_configure('gray', background='#cccccc')
        self.tree.tag_configure('white', background='#ffffff')

    def check_terminate_timeout(self):
        if self.terminating:
            return

        if self.last_empty_time is None:
            return

        if time.time() - self.last_empty_time >= self.terminate_on_empty_timeout:
            self.terminating = True
            self.destroy()
        else:
            self.after(1000, self.check_terminate_timeout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_as_daemon', action='store_true')
    args = parser.parse_args()

    if args.start_as_daemon:
        # Run Tk App in the main process thread and HTTPD in a worker thread
        window = App()
        try:
            with PipelineServer((SERVER_NAME, SERVER_PORT), RequestHandler, window=window) as httpd:
                httpd_thread = threading.Thread(target=httpd_main, args=(httpd,))
                httpd_thread.start()
                try:
                    window.mainloop()
                finally:
                    window.terminating = True
                    httpd.shutdown()
                    httpd_thread.join()
        except OSError as e:
            if e.errno == {'Windows': 10048, 'Linux': 98, 'Darwin': 48}[platform.system()]:
                # [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted
                # [Errno 98] Address already in use
                # [Errno 48] Address already in use
                print('Pipeline server socket address is already in use.')
                print('--- Exiting this pipeline server instance ---')
            else:
                raise
    else:
        # Create an independent / daemon process by starting a child process without inheriting
        # standard I/O streams and then immediately exiting this process to make it an orphan.
        subprocess.Popen([SYS_EXECUTABLE, PIPELINE_SERVER_FILEPATH, '--start_as_daemon'],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
