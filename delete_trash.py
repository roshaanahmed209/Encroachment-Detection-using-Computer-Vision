import os

def delete_all_files_recursive(folder_path):
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Folder not found: {folder_path}")

def main():
    folders = [
        r"H:\fyp\from_web_1",
        r"H:\fyp\matched",
        r"H:\fyp\resultant"
    ]
    for folder in folders:
        delete_all_files_recursive(folder)

if __name__ == "__main__":
    main()
