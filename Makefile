.PHONY: load_data
load_data:
	cd data && make final_task && cd ..

.PHONY: calculate_audio_embeddings
	python src/audio_embeddings.py

.PHONY: calculate_video_embeddings
	python src/video_embeddings.py