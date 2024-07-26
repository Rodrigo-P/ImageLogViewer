# ImageLogViewer Usage

## Data structure

Each well should have its own folder, with the imagelog CSVs and annotation placed inside following this folder structure:

|(Well_Name)/  
|---|IMLOG.csv  
|---|ESTATICO.csv  
|---|ACUSTICO.csv  
|---|class_data.json  
|---|labels/  
|---|---|000000.txt  
|---|---|000001.txt  
|---|---|000002.txt  
|---|---|...  

IMLOG.csv is the main log which will be processed with the custom windowing and run through the classification and segmentation models.
ESTATICO.csv is optional, an estatic version of the ´IMGLOG.csv´ and expected to have the same dimensions as the main file.
ACUSTICO.csv is optional, an acoustic version of the ´IMGLOG.csv´.

class_data.json contains annotated class data, used for the 'annotated' classification option.

The labels contains annotated bounding box structure data.

## Running on Windows

### Necessary programs

[Docker](https://docs.docker.com/desktop/install/windows-install/)

[VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/)
When launching VcXsrv, disable access control in extra settings.

### Build

The project image can be built with a standard command executed from the project folder:

```bash
docker build -t imagelogviewer .
```

### Run

To run the project it is necessary to setup a docker volume and obtain your IPv4.

The docker image doesn't have access to local files. To visualize your data create a docker volume and upload the imagelog CSVs to it.

To obtain the IPv4, run ipconfig in a terminal and it will be listed alongside other values. The IPv4 allows TKinter to display the interface from the docker into Windows through the VcXsrv.


```bash
docker run -ti --rm -e DISPLAY=[IPv4]:0.0 --volume [VolumeName]:/app/data imagelogviewer
```


## Running on Linux

### Data loading

The interface will have access to local files, but putting the imagelogs in the ´data´ folder makes for easier access.

### Create environment

```bash
python3.10 -m venv env
source env/bin/activate
```

### Install necessary packages

```bash
pip install -r "requirements.txt"
```

### Configure .env

```bash
python setup.py
```

### Run

```bash
python main.py
```
