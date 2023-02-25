accelerate launch --num_cpu_threads_per_process $1 $2 ${@:3}

find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
