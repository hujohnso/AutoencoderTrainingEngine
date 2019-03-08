#Current Project Description:

The project within this repository is currently for training many different types of autoencoders on images.  
Although any data set could be used the intended purpose is to train off of images obtained from YouTube videos.  
The current network topologies within this project that are ready to use are fully connected autoencoders,
convolutional autoencoders, fully convolutional autoencoders, and unet.  The training engine is extremely configurable
and allows for quickly changing hyper-parameters, data sets, and network topologies.

#Steps to pull and prepare code for running:
-Step 1: Clone the project to the location of your choice git clone https://github.com/hujohnso/AutoencoderTrainingEngine.git

-Step 2: Set up a virtual environment with python 3.6

-Step3 : Install all the necessary packages by running 'pip install -r requirements.txt from your terminal.

#Steps to pull and prepare code for running in pycharm:

The following is a description of how to run in pycharm but should contain the necessary steps to get this code running anywhere.

-Step 1: Download PyCharm: Visit https://www.jetbrains.com/pycharm/download/#section=linux and of course press download.

-Step 2: Extract pycharm-commuity and run.  cd into pycharm-community-2018.3.4/bin/ and run ./pycharm.sh

-Step 3: Now clone the project to the location of your choice "git clone https://github.com/hujohnso/AutoencoderTrainingEngine.git"

-Step 4: Open top level folder in the cloned directory 'research' in pycharm as the root of a existing project.

-Step 5: Set your project interpreter to python3.6 file->settings->project: research->ProjectInterpreter-> set the Project Intrepreter dropdown to python3.6

-Step6: Add a virtual environment to the pycharm project. file->settings->project: research->ProjectInterpreter->click the little gear next to the Project Intrepreter Dropdown and click 'new' then ok.
``
-Step7: Install all the necessary packages by running 'pip install -r requirements.txt from your terminal.

#Data Set Retrieval:

To train you must first retrieve the images necessary.  These are retrieved by running
 the main function in FrameRunner/FrameExtract.py
 
#Hyper-parameters and configuration:

Configuration for training models is in ModelRunner/ModelHyperParameters.py  Within this file there is a parent that 
sets up all the default hyper-parameters and configuration.  This class is named ModelHyperParameters and also contains
comments which describe how to use each of the parameters.  The child classes override the parent classes and allow
for easy storing of hyper-parameters for different data sets.

#Training a model:

To train a model go to the main.py at the root of the project.  There is one important function called
 run_all_steps(auto_encoder, hyper_parameters).  This function takes in two parameters.  One, auto_encoder, in this example
 which is the auto_encoder which you wish to train.  Note all network topologies inherit from the parent class AutoEncoder.py
 so all topologies will train using the method.  The second is the hyper_parameters class which you wish to use.  Running
 the main function will train the network and output the results to the Results/<the name of the results folder you specificy in
 the hyper-parameter class>.  
 


