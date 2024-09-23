import pandas as pd

def extract_starpep_data():
    # Load data from file
    with open('../data/general_peptides.fasta') as f:
        data = f.readlines()

    # Extract data
    starpep_data = []
    for line in data:
        if not line.startswith('>'):
            starpep_data.append(line.strip())

    df = pd.DataFrame(starpep_data, columns=['sequence'])
    df = df[df['sequence'].str.len() < 36]
    df = df[~df['sequence'].str.contains('[^ACDEFGHIKLMNPQRSTVWY]')]
    df = df[~df['sequence'].str.contains('X')]
    df = df[~df['sequence'].str.contains('Z')]
    df = df[~df['sequence'].str.contains('B')]
    df = df[~df['sequence'].str.contains('J')]
    df = df[~df['sequence'].str.contains('U')]
    df = df[~df['sequence'].str.contains('O')]
    df = df[~df['sequence'].str.contains('-')]

    df.to_csv("../data/starpep_sequences.csv", index=False)



if __name__ == '__main__':
    extract_starpep_data()