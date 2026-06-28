# s.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # Define dataset URL and paths
# local_zip = "fruits-360-original-size.zip"
# extract_dir = "fruits-360-original-size"
# kaggle_dataset = "moltean/fruits"
#
#
# def download_dataset_from_kaggle(dataset_name, output_file):
#     """Downloading dataset from Kaggle using kaggle API or fallback to manual instructions."""
#     try:
#         import kaggle
#         print(f"Downloading dataset from Kaggle: {dataset_name}...")
#         kaggle.api.authenticate()
#         kaggle.api.dataset_download_files(dataset_name, path='.', unzip=False)
#
#         downloaded_file = "fruits.zip"
#         if os.path.exists(downloaded_file):
#             os.rename(downloaded_file, output_file)
#             print("Download completed successfully.")
#             return True
#         else:
#             print("Download completed but file not found.")
#             return False
#
#     except ImportError:
#         print("Kaggle API not installed.")
#         print("\n=== MANUAL DOWNLOAD REQUIRED ===")
#         print("1. Go to: https://www.kaggle.com/datasets/moltean/fruits")
#         print("2. Click the 'Download' button (you may need to sign in)")
#         print(f"3. Save the downloaded file as '{output_file}' in this directory:")
#         print(f"   {os.getcwd()}")
#         print("\nThen run this script again.")
#         return False
#     except OSError as e:
#         if "credentials" in str(e).lower():
#             print("Kaggle credentials not found.")
#             print("\n=== MANUAL DOWNLOAD REQUIRED ===")
#             print("1. Go to: https://www.kaggle.com/datasets/moltean/fruits")
#             print("2. Click the 'Download' button (you may need to sign in)")
#             print(f"3. Save the downloaded file as '{output_file}' in this directory:")
#             print(f"   {os.getcwd()}")
#             print("\nThen run this script again.")
#             return False
#         raise
#     except Exception as e:
#         print(f"Failed to download: {str(e)}")
#         print("\n=== MANUAL DOWNLOAD REQUIRED ===")
#         print("1. Go to: https://www.kaggle.com/datasets/moltean/fruits")
#         print("2. Click the 'Download' button (you may need to sign in)")
#         print(f"3. Save the downloaded file as '{output_file}' in this directory:")
#         print(f"   {os.getcwd()}")
#         print("\nThen run this script again.")
#         return False
#
#
# def extract_zip_in_chunks(zip_file, extract_to, batch_size=2000):
#     """
#        Extracting a large zip file in chunks to avoid memory bottlenecks.
#        Processes a specified number of files (batch_size) at a time.
#        """
#     print("Extracting the dataset in chunks...")
#     os.makedirs(extract_to, exist_ok=True)
#
#     with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#         files = zip_ref.namelist()
#         total_files = len(files)
#
#         for i in range(0, total_files, batch_size):
#             batch = files[i:i + batch_size]
#             for file in batch:
#                 zip_ref.extract(file, extract_to)
#             print(f"Extracted {min(i + batch_size, total_files)} of {total_files} files...")
#     print(f"Dataset successfully extracted to '{extract_dir}'.")
#
#
# def is_valid_zipfile(zip_file):
#     """Check if a file is a valid zip file."""
#     try:
#         with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#             return zip_ref.testzip() is None
#     except (zipfile.BadZipFile, FileNotFoundError):
#         return False
#
#
# if __name__ == "__main__":
#     # Download the dataset if not already downloaded
#     if not os.path.exists(local_zip) or not is_valid_zipfile(local_zip):
#         if os.path.exists(local_zip):
#             print("Existing file is corrupted. Re-downloading...")
#             os.remove(local_zip)
#         success = download_dataset_from_kaggle(kaggle_dataset, local_zip)
#         if not success:
#             print("Exiting due to download failure.")
#             exit(1)
#     else:
#         print("Dataset already downloaded.")
#
#     # Extract the dataset if not already extracted
#     if not os.path.exists(extract_dir):
#         extract_zip_in_chunks(local_zip, extract_dir)
#     else:
#         print("Dataset already extracted.")
#
#     # Optional cleanup of the zip file
#     if os.path.exists(local_zip):
#         os.remove(local_zip)
#         print(f"Cleaned up zip file: {local_zip}")