SHELL=/bin/zsh

init:
	mkdir ./data
	touch ./data/.gitkeep

run:
	python main.py -c ./config/shapenet_ect_config.json

clean:
	rm -r **/processed

zip:
	echo "Zipped"
