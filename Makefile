reqs:
	poetry export -f requirements.txt --without-hashes > requirements.txt

mamba:
	pip install causal-conv1d>=1.1.0
	pip install mamba-ssm
	echo /usr/lib64-nvidia/ > /etc/ld.so.conf.d/libcuda.conf; ldconfig
