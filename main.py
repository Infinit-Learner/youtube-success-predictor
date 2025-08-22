from src.modeling.modeling import run_modeling, run_hyperparameter_tuning, run_interpretation,  save_artifacts, start_modeling, baseline_model_evaluation, model_selection, tuned_model_evaluation 
from src.preprocessing.preprocessing import run_preprocessing
import typer
app = typer.Typer()

@app.command()
def preprocessing(raw_path: str, cat_id_path: str ,processed_path: str):
    run_preprocessing(raw_path, cat_id_path,processed_path )


@app.command()
def modeling(training_path: str, test_path: str, metric: str = 'mse'):
    run_modeling(training_path, test_path, metric)

@app.command()
def run_pipeline(raw_path: str, cat_id_path: str ,processed_path: str, metric: str = 'mse'):
    run_preprocessing(raw_path, cat_id_path,processed_path )
    run_modeling(training_path=  f'{processed_path}/training_data.csv', 
                 test_path = f'{processed_path}/test_data.csv', metric = metric)

if __name__ == "__main__":
    app()