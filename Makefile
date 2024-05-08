DEVICE = cpu
PORT = 8888
GPU = all

install:
	pip install -r requirements-$(DEVICE).txt

build_image:
	docker build -t representation_learning -f $(DEVICE).dockerfile .

run_container:
	docker run --rm --gpus=$(GPU) -p $(PORT):8888 -v `pwd`:/assignment --name representation_learning representation_learning
