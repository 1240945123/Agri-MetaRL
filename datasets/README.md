# Datasets

Weather CSV files are stored locally under:

```text
datasets/weather/<location>/<year>.csv
```

The environment configuration currently expects locations such as `Amsterdam`, `Bleiswijk`, and `Spain`. Each yearly CSV must contain the columns expected by `gl_gym.environments.utils.load_weather_data`.

The weather collection is intentionally excluded from Git because the local dataset is roughly 200 MB. The repository reorganization preserved the existing files in `datasets/weather/`; it did not download, regenerate, or modify them. No download URL is documented because the repository does not contain an authoritative source declaration.
