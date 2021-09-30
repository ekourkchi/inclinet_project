
## clean-up, removing the products of the past executions
rm -rf compressed
rm -rf samples
rm -rf augmented
rm -rf models
mkdir samples
mkdir compressed
mkdir augmented

## resizing
python data_prep.py -p ./galaxies/ -b g -c catalogs/catalog.csv -s 128   -v
python data_prep.py -p ./galaxies/ -b r -c catalogs/catalog.csv -s 128   -v
python data_prep.py -p ./galaxies/ -b i -c catalogs/catalog.csv -s 128   -v
python data_prep.py -p ./galaxies/ -b gri -c catalogs/catalog.csv -s 128   -v


## data compression (generating npz files)
python data_compress.py -i 128x128_g -o compressed -s 128 -v
python data_compress.py -i 128x128_r -o compressed -s 128 -v
python data_compress.py -i 128x128_i -o compressed -s 128 -v
python data_compress.py -i 128x128_RGB -o compressed -s 128 -v

## train/test splits
## n is the number of data (sub-data) sets
python data_split.py -i ./compressed/ -o samples/ -n 3 -v

## data augmentation (making the sample uniform over the full range of inclinations)
## add "-z 1000" in production phase
## b is the number of traning batches 
## colorful and grascale images are combine in each training batch
python data_augment_uniform.py -i samples -o augmented -s 128 -n 3 -b 5 -z 3 -v

## Batch Training
python batch_training_regr.py -a augmented -i samples -s 128 -o models -m model4 -n 3 -b 5 -N 1 -v
python batch_training_regr.py -a augmented -i samples -s 128 -o models -m model5 -n 3 -b 5 -N 1 -v
python batch_training_regr.py -a augmented -i samples -s 128 -o models -m model6 -n 3 -b 5 -N 1 -v

