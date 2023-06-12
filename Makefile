SHELL=/bin/zsh

init:
	mkdir -p .venv
	python3.10 -m venv .venv
	pip install -r requirements.txt


bla:
	source .venv/bin/activate;
	pip install -r requirements.txt


run:
	python generate_experiments.py
	python main.py 

test_main:
	python generate_experiments.py
	python test_main.py 

clean:
	rm -r **/processed

zip:
	echo "Zipped"
