"""
Pipeline server used by stages of the processing pipeline for
https://github.com/mcmhsieh/Smile to display their working states and provide
information for each stage to wait for their predecessor to complete.

SPDX-FileCopyrightText: 2026 Mark Hsieh
SPDX-License-Identifier: MIT
"""

import sys
import argparse
import pathlib
import subprocess
import threading
import queue
import functools
import socket
import http.server
import time
import json
import requests
import tkinter as tk
from tkinter import ttk


PIPELINE_SERVER_FILEPATH = __file__
# Note that localhost is slow to resolve
SERVER_NAME = '127.0.0.1'
SERVER_PORT = 8081

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
            if len(self.server.pipeline_queue) == 0:
                self.server.window.schedule_destroy()
        self.server.window.schedule_update(self.server.pipeline_queue)
        self.send_response(200)
        self.end_headers()

def httpd_main(httpd):
    try:
        httpd.serve_forever()
    finally:
        window.schedule_destroy()

def start_pipeline_server():
    # Create an independent / daemon process by starting a child process and exiting this parent process.
    # For some reason, creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    # still creates a child process within the process tree.
    subprocess.Popen([sys.executable, PIPELINE_SERVER_FILEPATH])
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
        self.terminating = False

        self.title('Pipeline Server')
        self.geometry('500x300+100+100')

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

        self.tree.delete(*self.tree.get_children())

        for row_idx, (workspace_stage, state) in enumerate(pipeline_queue.items()):
            row_tag = ['gray', 'white'][row_idx % 2]
            self.tree.insert('', tk.END, values=(workspace_stage, state), tag=row_tag)
        self.tree.tag_configure('gray', background='#cccccc')
        self.tree.tag_configure('white', background='#ffffff')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_as_daemon', action='store_true')
    args = parser.parse_args()

    if args.start_as_daemon:
        # Run Tk App in the main process thread and HTTPD in a worker thread
        window = App()
        with PipelineServer((SERVER_NAME, SERVER_PORT), RequestHandler, window=window) as httpd:
            httpd_thread = threading.Thread(target=httpd_main, args=(httpd,))
            httpd_thread.start()
            try:
                window.mainloop()
            finally:
                window.terminating = True
                httpd.shutdown()
                httpd_thread.join()
    else:
        subprocess.Popen([sys.executable, PIPELINE_SERVER_FILEPATH, '--start_as_daemon'])
