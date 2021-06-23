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
