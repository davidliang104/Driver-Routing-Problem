import pandas as pd

def read_excel():
    read_workers()

def read_workers():
    # Read the Excel file
    df = pd.read_excel('Workers.xlsx')

    print(df)

def main():
    read_excel()

if __name__ == "__main__":
    main()