reqs:
	poetry export -f requirements.txt --without-hashes > requirements.txt

publish:
	poetry publish --build -u __token__ -p ${PYPI_TOKEN}

install:
	pip install -e .
	pip install hf_transfer
	apt install neovim btop -y
