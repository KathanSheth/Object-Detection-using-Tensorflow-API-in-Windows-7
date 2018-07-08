# Object-Detection-using-Tensorflow-API-in-Windows-7
# Object Detection Implementation on Oxford Pet Dataset in Windows 7

This was my first implementation of object detection. In parallel with this, I have implemented same project on my mac. 

The main problem between these two implementation were path declaration, download data and other required packages from internet.

Before implementing this, I have studied and read R-CNN, Fast/Faster R-CNN papers and watched/read videos/articles on YOLO and SSD.

There are many steps which need to performed to get it worked. Usually, my approach is to divide the problem in different parts/sub-topics and define an end-to-end pipeline (in my mind) so that it makes sense to me while implementation. Object Detection problem was little bit confusing to me because I was (am) not able to make connections between theoretical implementation of different architecture and practical implementation in Tensorflow API. After reading many articles, now I have some knowledge about implementation pipeline.

This implementation is based on the Tensorflow API example. I have implemented on my local machine.

Folder structure : 
```
+dataset
	+Oxford_pet
		-images
		-annotations
+training_data
	+Oxford_Pet
		-All tfrecord files
+frozen_models
	+Oxford_Pet
		+ssd_mobilenet_v1_coco_2018_01_28
			-All related files
+config_files
	+Oxford_Pet
		-ssd_mobilenet_v1_pets.config
+label_files
	+Oxford_Pet
		-pet_label_map.pbtxt
+train_eval_out
	+Oxford_Pet
		+train
		+eval
		+out
```

I have divided this project in steps. 

**Step 0.0** : Download tensorflow/models from Github (https://github.com/tensorflow/models/)

**Step 0.1** : Protobuf compilation

The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled.

This step is not as simple as it is mentioned in Tensorflow tutorial. There might be many ways to solve this.
But following is my hack.

First download protoc-3.4.0-win32 from internet and place it in Program Files folder.

Then from models/reseach folder:

Type `"C:\Program Files\protoc-3.4.0-win32\bin\protoc.exe" object_detection/protos/*.proto --python_out=.`

**Step 0.2** : Add libraries to PythonPath

Again there is minor change from the tutorial.

`set PYTHONPATH=PATH_TO_models\models;PATH_TO_models\models\research;PATH_TO_models\models\research\slim`

**Remember to run above command every time we open terminal**

Or set this path in environment variable permanently. I am too lazy to do that.!!

**Step 1.0** : Download the dataset to appropriate path

Here, I am working on Oxford-IIIT pet dataset. We have to download both image dataset and groundtruth dataset.

I have downloaded it from [here](http://www.robots.ox.ac.uk/%7Evgg/data/pets/) 

keep it in `dataset\Oxford_Pet` folder. (Not included in this repo as it is too big!)

**Step 2.0** : Conversion of Raw data to TFRecord format as Tensorflow Object Detection API expects any data to be converted to TFRecord first. 

In /models/research/dataset_tools/ there are many conversion scripts are provided. Tensorflow team has created these scripts for well-known dataset. If we have our own data then we need to write this conversion scripts on our own. 
I might need to create this type of script in future as I will be training custom object detector in future.


`python object_detection\dataset_tools\create_pet_tf_record.py \
    --label_map_path=object_detection\label_files\pet_label_map.pbtxt \
    --data_dir=object_detection\dataset\Oxford_Pet \
    --output_dir=object_detection\training_data`

 Two important things to keep in mind from above command:
 	1.	create_pet_tf_record.py creates tfrecords file for train and valid data. Data splitting is done in script.
 	2.	pet_label_map.pbtxt is protobuf file in text (Human readable format). This can be .pb (binary). Basically this file contains class id and class names.

 More details : https://www.tensorflow.org/extend/tool_developers/

 Here one more thing to notice. According to tutorial, there should be one train and one validation tfrecord files. But when we actually runs the script, it creates multiple record files for train and valid. 

 Reason : It's an improvement against creating a single file as this file will be loaded in RAM. This file will be in GB for some dataset. So it is better to create multiple files just like this example when trying on our own dataset (own script to generate records).

 As we have multiple record files, we have to modify config file (next) a little bit.
 Links to feed more than one tfrecord files:

 https://github.com/tensorflow/models/issues/3031

Keep this in `training_data\Oxford_Pet` folder. It should generate ten train and ten val files. (Not included in this repo as it is big!)

**Step 3.0** : Downloading a COCO pre-trained model for transfer learning

 Training a state of the art object detector from scratch can take days, even when using multiple GPUs! In order to speed up training, we'll take an object detector trained on a different dataset (COCO), and reuse some of it's parameters to initialize our new model.

 Some details about this download folder:
 
 This folder contails following important files:

	-	a graph proto (graph.pbtxt)
	-	a checkpoint (model.ckpt.data-00000-of-00001, model.ckpt.index, model.ckpt.meta)
	-	a frozen graph proto with weights baked into the graph as constants (frozen_inference_graph.pb) to be used for out of the box inference 

Download `ssd_mobilenet_v1_coco_2018_01_28` from [Tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

Keep the `ssd_mobilenet_v1_coco_2018_01_28` in `frozen_models\Oxford_Pet` folder. (Not included here as it is big!)
	
**Step 4.0** : Configuring the object detection pipeline

In the Tensorflow Object Detection API, the model parameters, training parameters and eval parameters are all defined by a config file.
The Tensorflow Object Detection API uses protobuf files to configure the training and evaluation process. 
At a high level, the config file is split into 5 parts:
1.	The model configuration. This defines what type of model will be trained (ie. meta-architecture, feature extractor).
2.	The train_config, which decides what parameters should be used to train model parameters (ie. SGD parameters, input preprocessing and feature extractor initialization values).
3.	The eval_config, which determines what set of metrics will be reported for evaluation.
4.	The train_input_config, which defines what dataset the model should be trained on.
5.	The eval_input_config, which defines what dataset the model will be evaluated on. Typically this should be different than the training input dataset.

More details : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md

So I have updated the config files accordingly. 
Kept all the train and val record files in a list and provided a path.

Keep config file `ssd_mobilenet_v1_pets.config` in ``config_files\Oxford_Pet` fodler

**Step 5.0** : Actual Training/Evaluation and Tensorboard

`python object_detection\train.py --logtostderr --pipeline_config_path=C:\Users\shk1ply\tensor\ObjectDetection\1_Basic_Implementation\models\research\object_detection\config_files\Oxford_Pet\ssd_mobilenet_v1_pets.config --train_dir=C:\Users\shk1ply\tensor\ObjectDetection\1_Basic_Implementation\models\research\object_detection\train_eval_out\Oxford_Pet\train`

**Here, I have used MobileNet with SSD because I was getting resourceexhausterror while using ResNet even with batch_size=1.
Even if MobileNet, I was getting resourceexhausterror for batch_size=24. So I had to reduce it to 8.
MobileNet is fast but I have to compromise with accuracy. Considering this, I am expecting okayish performance with this model as we have 37 classes.
But the whole idea is to set the working pipeline on Windows and to understand the process.
Once everything is up, I will move forward for my custom dataset on Amazon AWS.**

Try to reduce the batch_size in case you get resourceexhausterror.

If you get NO Module name "xyz" found error, then try to set the above path from terminal.
It will solve this issue.

The other error which I faced : 

ValueError: Tried to convert 't' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted [].
	Issue with Python version. 
	
	Solution is here : https://github.com/tensorflow/models/issues/3705#issuecomment-375563179
	
In parallel, open second terminal.

Set the path mentioned above and from model\reseach directory:

`python object_detection/eval.py --logtostderr --pipeline_config_path=C:\Users\shk1ply\tensor\ObjectDetection\1_Basic_Implementation\models\research\object_detection\config_files\Oxford_Pet\ssd_mobilenet_v1_pets.config --checkpoint_dir=C:\Users\shk1ply\tensor\ObjectDetection\1_Basic_Implementation\models\research\object_detection\train_eval_out\Oxford_Pet\train --eval_dir=C:\Users\shk1ply\tensor\ObjectDetection\1_Basic_Implementation\models\research\object_detection\train_eval_out\Oxford_Pet\eval`

Open third terminal:

Set the path and 

`tensorboard --logdir=C:\Users\shk1ply\tensor\ObjectDetection\1_Basic_Implementation\models\research\object_detection\train_eval_out\Oxford_Pet`

We can see loss, mAP etc and in Images tab, we can visualize bounding boxes.

I run training for ~20K steps. As expected, my loss was not going below 1.5
Stop training by CTRL + C.

**Step 6.0** Save Frozen Model

Once, we stop training we can see some model.ckpt files in our train folder.

Check if you have all three (meta, index, data) files for a step.

Now we have to export this as an inference graph and for that we use `export_inference_graph.py` file which is bundled in Tensorflow API.

I am going to save it in train_eval_out\Oxford_Pet\out directory.

`python export_inference_graph.py --input_type image_tensor --pipeline_config_path Path_to_models\models\research\object_detection\config_files\Oxford_Pet\ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix Path_to_models\models\research\object_detection\train_eval_out\Oxford_Pet\train/model.ckpt-19826 --output_directory Path_to_models\models\research\object_detection\train_eval_out\Oxford_Pet\out`

Check `out` directory and confirm that we have `frozen_inference_graph.pb` file.

Now it's time to check how our model works on test images.

**Step 7.0** Evaluate the model on test data.

For evaluation, we use `object_detection_tutorial.ipynb` which is bundled with Tensorflow API and is in object_detection folder.

So from terminal model\reseach\object_detection

`jupyter notebook object_detection_tutorial.ipynb`

Here we have to make few modification. I have downloaded few images from internet and
kept them in test_images folder.

Conclusion : As expected, it is working okayish!! It is not detecting due to low confidence and there are few wrong detection also!

To-Do:

1.	Object Detection with Custom data
2.	Object Detection on Video for Self driving car.
3.	Implement on Raspberry Pi
