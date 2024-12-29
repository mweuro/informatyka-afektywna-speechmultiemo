GIT_REPO = https://github.com/CheyneyComputerScience/CREMA-D.git
REMOTE_AUDIO = AudioWAV
REMOTE_VIDEO = VideoFlash
REMOTE_ACTORS_META = VideoDemographics.csv
REMOTE_FILENAMES = SentenceFilenames.csv
LOCAL_AUDIO = audio_data
LOCAL_VIDEO = video_data
LOCAL_ACTORS_META = actors_meta.csv
LOCAL_RAW_DATA = emotions.csv
RAW_DIR = raw
TEMP_DIR = temp
ALIAS_SCRIPT = ../src/load_data/raw_data_aliases.py

.PHONY: load
load:
	mkdir $(RAW_DIR) && \
	git clone --depth 1 --no-checkout $(GIT_REPO) $(TEMP_DIR) && \
	cd $(TEMP_DIR) && \
	git sparse-checkout init --cone && \
	git sparse-checkout set $(REMOTE_AUDIO) $(REMOTE_VIDEO) $(REMOTE_ACTORS_META) $(REMOTE_FILENAMES) && \
	git checkout && \
	mv $(REMOTE_AUDIO) ../$(LOCAL_AUDIO) && \
	mv $(REMOTE_VIDEO) ../$(LOCAL_VIDEO) && \
	mv $(REMOTE_ACTORS_META) ../$(LOCAL_ACTORS_META) && \
	mv $(REMOTE_FILENAMES) ../$(RAW_DIR)/$(LOCAL_RAW_DATA) && \
	cd .. && \
	rm -rf $(TEMP_DIR)

add_aliases:
	python $(ALIAS_SCRIPT)

final_task:
	make load && make add_aliases