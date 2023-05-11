init:
	mkdir ./data
	touch ./data/.gitkeep

run:
	python main.py -c config.json

clean:
	echo "Cleaned"

zip:
	echo "Zipped"
