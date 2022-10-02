import os
import glob
def get_job_id(dir_path):
    files = glob.glob(f"{dir_path}/*.txt")
    if len(files) == 0:
        return 1
    else:
        file_names = [os.path.splitext(os.path.basename(file))[0] for file in files]
        job_ids = [int(name) for name in file_names if name.isdigit()]
        return max(job_ids) + 1

print(get_job_id("./jobs"))
