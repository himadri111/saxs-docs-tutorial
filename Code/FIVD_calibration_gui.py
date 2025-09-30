# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 11:19:25 2025

@author: Himadri
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# ---------- Defaults ----------
DEFAULTS = {
    "start_diff": "152",
    "og_shape": "2560,2560",
    "top_path": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded",
    "path": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/region_analysis/left_lateral",
    "calibPath": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/CT data/calibration/",
    "lowPath": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/CT data/low res/inverse scaled/",
    "highPath": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/CT data/high res/inverse scaled/",
    "beta_path": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/region_analysis/left_lateral/167208_phi_5x5_dilated",
    "alpha_path": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/region_analysis/left_lateral/167208_theta_5x5_dilated",
    "index_path": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/region_analysis/left_lateral/167208_fibreID_5x5_dilated",
    "pad_file": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/vox_padding_fivd1.xlsx",
    "ct_savePath": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/CT data/calibrated/",
    "fibtrac_savePath": "/dls/i13/data/2023/sm29784-9/processing/tomoSAXS/FIVD_1/new_recon/unloaded/region_analysis/left_lateral",
    "scan_name": "FIVD1_unloaded_new",
}

# Keys that should get a directory picker
DIR_KEYS = {
    "top_path", "path", "calibPath", "lowPath", "highPath",
    "beta_path", "alpha_path", "index_path", "ct_savePath", "fibtrac_savePath"
}
# Key that should get a file picker
FILE_KEY = "pad_file"


class ConfigApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Config Input")

        self.vars = {}
        self.values = {}

        container = ttk.Frame(self, padding=10)
        container.pack(fill="both", expand=True)

        # Build rows
        for row, (key, default) in enumerate(DEFAULTS.items()):
            ttk.Label(container, text=key).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 6))

            var = tk.StringVar(value=default)
            entry = ttk.Entry(container, textvariable=var, width=70)
            entry.grid(row=row, column=1, sticky="ew", pady=2)
            self.vars[key] = var

            # Add Browse… button where appropriate
            if key in DIR_KEYS:
                ttk.Button(container, text="Browse…", command=lambda v=var: self.browse_dir(v)).grid(
                    row=row, column=2, sticky="w", padx=6
                )
            elif key == FILE_KEY:
                ttk.Button(
                    container, text="Browse…",
                    command=lambda v=var: self.browse_file(v, (("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")))
                ).grid(row=row, column=2, sticky="w", padx=6)

        container.columnconfigure(1, weight=1)

        # Confirm button
        ttk.Button(self, text="Confirm", command=self.confirm).pack(pady=8)

        # Make the window hug its contents
        self.update_idletasks()
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")

    def browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(title="Select directory", initialdir=var.get() or "/")
        if path:
            var.set(path)

    def browse_file(self, var: tk.StringVar, filetypes):
        path = filedialog.askopenfilename(title="Select file", filetypes=filetypes, initialdir=var.get() or "/")
        if path:
            var.set(path)

    def confirm(self):
        self.values = {k: v.get() for k, v in self.vars.items()}  # all strings
        messagebox.showinfo("Saved", "Values saved into memory (self.values).")
        self.destroy()


def main():
    app = ConfigApp()
    app.mainloop()
    print("Collected values:")
    for k, v in app.values.items():
        print(f"{k} = {v!r}")
        
def run_gui():
    app = ConfigApp()
    app.mainloop()
    # After window closes, values are available:
    print("Collected values:")
    for k, v in app.values.items():
        print(f"{k} = {v!r}")
    return app
