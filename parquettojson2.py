import pandas as pd
import json
import os
from tkinter import Tk, filedialog, messagebox, Label, Button, Listbox, Scrollbar, SINGLE, END

class ParquetFolderToJsonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Parquet Folder to JSON Converter")

        # UI Elements
        Label(root, text="Parquet Files in Folder:").pack()

        self.file_listbox = Listbox(root, selectmode=SINGLE, width=60, height=15)
        self.file_listbox.pack()

        scrollbar = Scrollbar(root, command=self.file_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        Button(root, text="Select Folder", command=self.load_folder).pack(pady=5)
        Button(root, text="Convert to JSON", command=self.export_all_to_json).pack(pady=5)

        self.folder_path = None
        self.parquet_files = []

    def load_folder(self):
        # Open folder dialog
        folder_selected = filedialog.askdirectory()
        if not folder_selected:
            return

        self.folder_path = folder_selected
        self.parquet_files = [f for f in os.listdir(self.folder_path) if f.endswith(".parquet")]

        self.file_listbox.delete(0, END)
        for file in self.parquet_files:
            self.file_listbox.insert(END, file)

        messagebox.showinfo("Success", f"Loaded {len(self.parquet_files)} Parquet files from:\n{self.folder_path}")

    def export_all_to_json(self):
        if not self.folder_path or not self.parquet_files:
            messagebox.showwarning("Warning", "No folder selected or no Parquet files found!")
            return

        for file in self.parquet_files:
            file_path = os.path.join(self.folder_path, file)

            try:
                # Load the Parquet file into a DataFrame
                df = pd.read_parquet(file_path)

                # Check if 'conversation' column exists
                if 'conversations' not in df.columns:
                    messagebox.showwarning("Warning", f"Skipping {file}: No 'conversation' column found.")
                    continue

                # Extract conversation column
                conversation_data = df[['conversations']]

                # Convert to JSON format
                json_data = conversation_data.to_json(orient="records", indent=4)

                # Save JSON file with same base name
                json_file_path = os.path.join(self.folder_path, file.replace(".parquet", ".json"))
                with open(json_file_path, "w", encoding="utf-8") as json_file:
                    json_file.write(json_data)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process {file}:\n{e}")
                continue

        messagebox.showinfo("Success", "All Parquet files have been converted to JSON!")

if __name__ == "__main__":
    root = Tk()
    app = ParquetFolderToJsonGUI(root)
    root.mainloop()
