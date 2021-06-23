# Deep Learning Final Competition
## Directory Structure
```
|- final_competition
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
	|- run.sh

```
## Enviroment and Installation Setups
- Python 3.8
- Pytorch 1.8,1
- `pip install -r requirements.txt`

## Dataset
- Download audio dataset from [link](https://www.kaggle.com/c/nycu-dl-final-competition/data)

## Run
- After entering directory '309555025', execute `run.sh`
```
$ ./run.sh
```

- Otherwise, execute following two commands respectively if you are on Windows
```
$ python audio_to_text.py
$ python text_to_label.py
```
## Model Architecture
### Audio to Text
- Whole Picture
```
	Audio => MelSpectrogram(n_mel=64) => 3 Layers of CNN => 3 Layers of RNN => CTC Loss => CTC GreedyDecoder => Text
``` 

- Audio process
	- Torchaudio.transforms.MelSpectrogram
- CNN
	- Channel: 32
	- Kernel: 3
	- Padding: 1
	- Stride: 2
- RNN
	- Input size: 512
	- Hidden size: 512
	- Bidirectional: True

- Hyperparameters
	- Batch size: 16
	- Learning rate: 0.001
	- Epochs: 50

### Text to Label
- Whole Picture
```
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
