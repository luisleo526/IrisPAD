accelerate launch --num_cpu_threads_per_process $1 main.py ${@:2}

find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
