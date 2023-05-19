SHELL=/bin/zsh

init:
	pip install poetry
	poetry config virtualenvs.in-project true 
	poetry install 
	poetry shell

activate:

run:
	python main.py 

clean:
	rm -r **/processed

zip:
	echo "Zipped"
