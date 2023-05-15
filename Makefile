SHELL=/bin/zsh

init:
	pip install poetry
	poetry config virtualenvs.in-project true 
	poetry install 
	poetry shell

activate:

run:
	python main.py -c ./config/gnn_mnist_ectlinear_config.json

clean:
	rm -r **/processed

zip:
	echo "Zipped"
