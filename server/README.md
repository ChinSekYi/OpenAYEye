# Dashboard Backend
To serve the app after the front end has been built,
you need to build the react app in `./dashboard`. 
## Build React app
To build the react dashboard, run
```bash
npm run build
```

## Setup
Cross over to this directory
```bash
cd OpenAYEye/server
```

In this directory, first create and install conda env,
```bash
conda env create -f environment.yml
```

Then enter conda environment
```bash
. activate flaskdev
```

## Testing & Development
Run
```bash
flask --app app.py run
```

Default link to site 
[http://localhost:5000](http://localhost:5000)