# Deep Learning Final Competition
## Directory Structure
```
|- 309555025
	|- test
		|- 000000.wav
		|- ...
	|- train
		|- 000000.wav
		|- ...
	|- audio_to_text.py
	|- text_to_label.py
	|- model.py
	|- utils.py 
	|- train.csv
	|- run.sh

```
## Enviroment and Installation Setups
- **Python 3.8**
- Pytorch 1.8.1
- Audio file I/O backend
	- Windows: `pip install PySoundFile`
	- Linux: `pip install sox`
- `pip install -r requirements.txt`

## Dataset
- Download audio dataset from [link](https://www.kaggle.com/c/nycu-dl-final-competition/data)

## Run
- After entering directory '309555025', execute `run.sh` (not recommend but it's simpler)
```
$ chmod +x run.sh
$ ./run.sh
```

- If `run.sh` fails, try to execute following two commands in order (better option)
```
$ python audio_to_text.py
$ python text_to_label.py
```
- Notifications
	- After successfully finish running `audio_to_text.py`, there should be a file `decoded_text.txt` for `text_to_label.py` to read
	- After successfully finish running `text_to_label.py`, there should be a file `submission.csv` for grading 
	- It might take couple hours to finish the whole process, you could shorten the process time by adjusting the variable `EPOCHS` in  `audio_to_text.py`
	- Recommend that the memory size of GPU should be larger or equal to 6GB


## Model Architecture
### Audio to Text
- Whole Picture
	- 	```
		Audio => MelSpectrogram(n_mel=64) => 3 Layers of CNN => 3 Layers of RNN => CTC Loss => CTC GreedyDecoder => Text
		```
- Audio process
	- Torchaudio.transforms.MelSpectrogram(sr=16000,n_mel=64)
- CNN
	- Channel: 32
	- Kernel: 3
	- Padding: 1
	- Stride: 2
- RNN
	- Input size: 512
	- Hidden size: 512
- Hyperparameters
	- Batch size: 16
	- Learning rate: 0.001
	- Epochs: 50

### Text to Label
- Whole Picture
	- 	```
		Text => Embedding(embed_size=512) => 1 Layers of RNN(Hidden size=256) => Cross Entropy Loss => Label
		``` 
- RNN
	- Input size: 512
	- Hidden size: 256
- Hyperparameters
	- Batch size: 16
	- Learning rate: 2.5
	- Epochs: 10
	- Embedding dimension: 512
	- Hidden size: 256