import pandas as pd
import yaml



YAML_PATH = '../params.yaml'
FILENAMES_PATH = 'raw/emotions.csv'
OUTPUT_PATH = 'raw/emotions.csv'


def load_yaml(file_path: str) -> dict[dict[str, str]]:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def add_meta(df: pd.DataFrame, vars: dict[dict[str, str]]) -> pd.DataFrame:
    df['Sentence'] = df['Filename'].apply(lambda file: vars['sentence'][file[5:8]])
    df['Emotion'] = df['Filename'].apply(lambda file: vars['emotion'][file[9:12]])
    df['EmotionLevel'] = df['Filename'].apply(lambda file: vars['emotion_level'][file[13:]])
    return df


def main() -> None:
    vars = load_yaml(YAML_PATH)
    df = pd.read_csv(FILENAMES_PATH)
    df = df.drop(columns = ['Stimulus_Number'], axis = 1)
    df = add_meta(df, vars)
    df.to_csv(OUTPUT_PATH, index = False, sep = ',')


if __name__ == '__main__':
    main()