# Verifying Access to the HPC System

This tutorial guides you through the steps to confirm that you can successfully access the HPC resources available for this course. If you encounter any issues logging into `greene` or `burst`, please contact the course staff immediately.

## Part 1: Logging into Greene

### Prerequisites
- Terminal app on Linux or Mac
- Command Prompt or an SSH client on Windows 10

### Steps
1. **Initial SSH into HPC Gateway (Optional)**: 
  - If you're already connected to the NYU Network or using the [NYU VPN](https://www.nyu.edu/life/information-technology/infrastructure/network-services/vpn.html), you can skip this step.
  - Otherwise, open your terminal or command prompt and run the following SSH command:
    ```bash
    ssh <NetID>@gw.hpc.nyu.edu # When prompted, enter the password associated with your NYU NetID.
    ```

2. **SSH into Greene**: 
  - Type the following command in the terminal and hit enter:
    ```bash
    ssh <NetID>@greene.hpc.nyu.edu
    ```

3. **Confirm Successful Login**: 
  - After running the above command, your terminal prompt should change to something like `[<NetID>@log-3 ~]$`.

## Part 2: Logging into Burst

### Steps
1. **SSH into Burst from Greene**: 
  - While logged into a `greene` node, execute the following command:
    ```bash
    ssh burst
    ```

2. **Confirm Successful Login**: 
  - If successful, your terminal prompt will change to something like `[<NetID>@log-burst ~]$`.

## Part 3: Exploring the Burst

### How to Run a CPU-only Interactive Job
- Execute the following command to initiate a simple CPU-only interactive job that lasts for 4 hours:
  ```bash
  srun --account=ds_ga_1011-2023fa --partition=interactive --time=04:00:00 --pty /bin/bash
         # It may take a few minutes so please be patient :)
  ```
- Your terminal prompt should change to something like `bash-4.4$`.
- To exit the job and return to your regular terminal, simply type `exit`.

### How to Run a GPU Job with 1 V100 GPU
- To launch a GPU job that utilizes one V100 GPU and lasts for 4 hours, execute:
  ```bash
  srun --account=ds_ga_1011-2023fa --partition=n1s8-v100-1 --gres=gpu:v100:1 --time=04:00:00 --pty /bin/bash
         # It may take a few minutes so please be patient :)
  ```

---

Feel free to reach out to the course staff for any further questions or clarifications.