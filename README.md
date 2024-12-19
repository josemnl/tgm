# Transitional Grid Map

## Clone repository
```
git clone https://gits-15.sys.kth.se/jmgs/TGMp
```

## Install requirements
```
pip install -r .\requirements.txt
```

## Train the network
```
python ./src/nuscenes-scenesCache.py
python ./src/nuscenes-datGen.py
```

## Run the network
Update the name of the network in `./src/nuscenes-run.py`. Then:

```
python ./src/nuscenes-run.py
```
