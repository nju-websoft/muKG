=================
Run models with args
=================

This is the cmd command code to train TransE. This code shows how to select tasks and training mode::

       PS D:\muKG> python main_args.py -t lp -m transe -o train -d data/FB15K

This is the cmd command code to train TransE with multi-GPU. This code shows how to choose GPU numbers and worker numbers::

       PS D:\muKG> python main_args.py -t lp -m transe -o train -d data/FB15K -r gpu:2 -w 2