import pandas as pd



def generate_fasta_file(file_path: str,
                        out_path: str):
    with open(file_path) as f:
        with open(file_path + ".fasta", "w") as out:
            for line in f.readlines():
                if line.startswith("sequence"):
                    pass
                else:
                    out.write(f">{line}")
                    out.write(line)



def clean_output_file(file_path: str):
    """
    Cleans the output file by removing empty lines and unnecessary characters.
    """
    count = 0
    with open(file_path, "r") as f:
        for line in f.readlines():
            field_12 = line.split("\t")[12]

            field_13 = line.split("\t")[13]
            field_14 = line.split("\t")[14].strip()

            if field_12.startswith("GO"):
                print(1)
            if not field_13.startswith("GO"):
                count += 1
            if field_14.startswith("GO"):
                print(3)
    print(count)
            # print(f"Field: 13 {field_12}")
            # print(f"Field: 14 {field_13}")
            # print(f"Field: 15 {field_14}")






if __name__ == "__main__":
    # generate_fasta_file(file_path="../../../data/train_data/mlm_starpep_data.csv",
    #                     out_path="../../../data/train_data/mlm_test.fasta")
    clean_output_file("../../../data/train_data/mlm/mlm_starpep_data_go_annotated.tsv")