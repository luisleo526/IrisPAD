rm -rf ./log
accelerate launch --num_cpu_threads_per_process $1 main.py
