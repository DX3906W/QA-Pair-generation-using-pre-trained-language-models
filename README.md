# QA-Pair-generation-using-pre-trained-language-models
Three different approaches were implemented and fine-tuned on the **SQuAD dataset** to generate QA pairs: 
* multi-task, 
* pipeline, 
* joint model. 

The **distractors** are essential for multi-choice questions, one **T5-based distractor generation model** is fine-tuned with the **DG-RACE dataset**. 

The **evaluation** stage includes two separate methods: 
* automatic evaluation: the metrics for the text generation models are employed, including BLEU, METEOR, and ROUGE.
* expert evaluation: involved semi-structured interview to carry out the qualitative analysis.

The structure of the code files:
```
│  convert_to_json.py   # Convert the benchmark data into json format
│  data_loader.py	   # Load data
│  test_model.py		   # Fine-tuned model testing, just for development
│  trainer.py		   # Trainer for all tasks
│  utils.py			   # Independent functions, including automatic evaluation metrics
│  
├─benchmark_dataset   # The folder to put benchmark dataset. *.txt is the created QA. *.json is the processed data, we should use this in the evaluation
│      Big Data.txt
│      big_data.json
│      EnterpriseModeling.txt
│      enterprise_modeling.json
│      python_programming.json
│      The Python 100.txt
│      
├─benchmark_qa		# The folder to put the generated QA from benchmark dataset
├─distractor			# Distrsctor generation model and datasets
│  │  data_processor.py		# Data preprocessing of distractor generation
│  │  model.py				# Dsitractor generation model
│  │  trainer.py			# The training functions of the distractor model
│  │  
│  ├─dataset				# The dataset for distractor generation, Recommend to use updated dataset.
│  │      race_dev_original.json
│  │      race_dev_updated.json
│  │      race_test_original.json
│  │      race_test_updated.json
│  │      race_train_original.json
│  │      race_train_updated.json
│  │      
│          
├─joint	
│  │  data_processor.py		# The data preprocessing for joint model
│  │  model.py 			# Definition of the joint model
│  │  trainer.py			# The training functions of the joint model
│  │  
│          
├─multitask
│  │  data_processor.py		# The data preprocessing for multi-task model
│  │  model.py				# Definition of the multi-task model
│  │  trainer.py			# The training functions of the multi-task model
│  │  
│          
├─pipeline
│  │  data_processor.py		# The data preprocessing for pipeline model
│  │  model.py				# Definition of the pipeline model
│  │  trainer.py			# The training functions of the pipeline model
│  │  
│          
├─processed_squad			# The preprocessed SQuAD dataset will be saved here
├─saved_models			# The folder to save the fine-tuned models
```


