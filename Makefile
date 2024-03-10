reqs:
	poetry export -f requirements.txt --without-hashes > requirements.txt

publish:
	poetry publish --build -u __token__ -p ${PYPI_TOKEN}
