Setting up environment:
Create a conda environment with required packages as mentioned in requirements.txt

The model can be trained using the following command.
python train_PED.py --model_name "finetune" --epochs 5 --batchsize 8

The code has been provided for one pathology, PED. For other pathologies IRF and SRF, code can be modified accordingly by replacing PED with corresponding pathology.
 
Under csv_files folder we have provided one instance in train.csv for reference. Similarly prepare test.csv

The pretrainednormal folder contains a normal classification model for PED pathology trained with cross-entropy loss which has to be fine-tuned using the approach given in the paper. The complete trained model for PED pathology is present in pretrainedmodel folder. 

The testing of model can be done with the following command.
python test.py --model_name "Small_Inception ResnetV2_with_pretrained_weights" --disease "PED" --weight_path "../fulltrainedmodels/model_PED_BestF1_finetune_dice_loss.pt"