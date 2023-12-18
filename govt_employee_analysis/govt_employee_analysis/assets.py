import json
import os
import pandas as pd
import requests
import csv
from io import BytesIO
from pymongo import MongoClient
from dagster import asset, AssetExecutionContext, MetadataValue, MaterializeResult
import psycopg2
import plotly.express as px
import base64
import numpy as np
import matplotlib.pyplot as plt
import pandas.io.sql as sqlio
import seaborn as sns
from sqlalchemy import create_engine,event,text,exc
from sqlalchemy.engine.url import URL

@asset
def JsonDownload() -> None:
    reimburesment_url = "https://data.cityofchicago.org/resource/g5h3-jkgt.json?$limit=7000"
    reimburesment_url_data = requests.get(reimburesment_url).json()

    os.makedirs("data", exist_ok=True)
    with open("data/employee_reimbursment.json", "w") as f:
        json.dump(reimburesment_url_data, f)

@asset (deps=[JsonDownload])
def ExtractingJson() -> None:
    
    client = MongoClient("mongodb://dap:dap@localhost:27017/")
    db = client['dap']

    collection = db['reimbursements']
    with open("data/employee_reimbursment.json") as erj:
        data = json.load(erj)
    for reimburesment in data:
        collection.insert_one(reimburesment)        

    client.close()

    print("Data loaded into MongoDB successfully!")

@asset (deps=[ExtractingJson])
def JsonLoading(context: AssetExecutionContext,
) -> pd.DataFrame:
    client = MongoClient("mongodb://dap:dap@localhost:27017/")
    db = client['dap']
    collection = db['reimbursements']
    data = list(collection.find())
    reimburesment_df = pd.DataFrame(data)
    reimburesment_df.columns = reimburesment_df.columns.str.lower()
    reimburesment_df = reimburesment_df.rename(columns={'vendor_name':'name'})
    context.add_output_metadata(
    metadata={
        "num_records": len(reimburesment_df),
        "preview": MetadataValue.md(reimburesment_df.head().to_markdown()),
    }
    )
    return reimburesment_df




@asset
def csvDownload() -> None:
    employee_url = "https://data.cityofchicago.org/resource/xzkq-xp2w.csv?$limit=40000"

    response = requests.get(employee_url)

    if response.status_code == 200:
        os.makedirs("data", exist_ok=True)

        with open("data/employee_details.csv", "w", newline="") as f:
            f.write(response.text)
            
        print("CSV file successfully created.")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")



@asset(deps=[csvDownload])
def ExtractingCsv() -> None:
    connection_string = "postgresql+psycopg2://dap:dap@127.0.0.1:5432/dap_project"

    table_name = 'employee_details'
    employee_work_details = pd.read_csv("data/employee_details.csv")
    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            connection.execution_options(isolation_level="AUTOCOMMIT")
            connection.execute(text(f"CREATE TABLE IF NOT EXISTS {table_name} ();"))
            employee_work_details.to_sql(table_name, connection, if_exists='replace', index=False)
    except exc.SQLAlchemyError as dbError:
        print("PostgreSQL Error", dbError)
    finally:
        if engine in locals():
            engine.close()

@asset(deps=[ExtractingCsv])
def LoadingCsv(
    context: AssetExecutionContext,
) -> pd.DataFrame:
    connection_string = "postgresql+psycopg2://dap:dap@127.0.0.1:5432/dap_project"
    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            employee_df = sqlio.read_sql_query(text("SELECT * FROM employee_details;"), connection)
            employee_df.columns = employee_df.columns.str.lower()
            context.add_output_metadata(
            metadata={
                "num_records": len(employee_df),
                "preview": MetadataValue.md(employee_df.head().to_markdown()),
            }
        )
        return employee_df
    except exc.SQLAlchemyError as dbError:
        print("PostgreSQL Error", dbError)
    finally:
        if engine in locals():
            engine.close()
    
@asset
def PercentageFullTimeAndPartTimePerDepartment(LoadingCsv) -> MaterializeResult:
    employee_df = LoadingCsv
    contingency_table = pd.crosstab(employee_df['department'], employee_df['full_or_part_time'], margins=True, margins_name="Total")
    proportions = contingency_table.div(contingency_table['Total'], axis=0).drop(columns='Total')
    proportions_reset = proportions.reset_index()

    departments = (proportions_reset['department'])
    employment_counts = {
        'F': np.array(proportions_reset['F']),
        'P': np.array(proportions_reset['P']),
    }
    width = 0.6  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots(figsize =(10,15))
    bottom = np.zeros(len(departments))

    for employment, employment_count in employment_counts.items():
        p = ax.bar(departments, employment_count, width, label=employment, bottom=bottom)
        bottom += employment_count

    plt.xticks(rotation=60, ha='right')
    ax.set_title('Number of Full Time or Part Time Employees by Department')
    ax.legend()
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format = "png", pad_inches=0)
    image_data = base64.b64encode(buffer.getvalue())

    md_content = f"![img](data:image/png;base64,{image_data.decode()})"

    return MaterializeResult(
        metadata={
            "plot": MetadataValue.md(md_content)
        }
    )


@asset
def SalaryDistribPerDept(LoadingCsv) -> MaterializeResult:
    employee_df = LoadingCsv
    average_salary_by_department = employee_df.groupby('department')['annual_salary'].mean().sort_values()
    average_hourly_rate_by_department = employee_df.groupby('department')['hourly_rate'].mean().sort_values()

    return MaterializeResult(
            metadata={
                "Department having highest annual salary:": MetadataValue.md(str(average_salary_by_department.max())),
                "Department having highest Hourly Rate:" : MetadataValue.md(str(average_hourly_rate_by_department.max()))
            }
        )

@asset
def Merged_data(context: AssetExecutionContext,JsonLoading,LoadingCsv) -> pd.DataFrame:
    reimburesment_df = JsonLoading
    employee_df = LoadingCsv
    data_df = pd.merge(reimburesment_df,employee_df,on=['name','department'], how = 'inner')
    context.add_output_metadata(
    metadata={
        "num_records": len(data_df),
        "preview": MetadataValue.md(data_df.head().to_markdown()),
    }
    )
    return data_df
    
@asset
def reimburesment_per_person(context: AssetExecutionContext, Merged_data) -> pd.DataFrame:
    data_df = Merged_data
    vouchers_per_person = data_df.groupby(['name','department'])['voucher_number'].nunique()
    vouchers_per_person = pd.DataFrame(vouchers_per_person)
    context.add_output_metadata(
    metadata={
        "num_records": len(vouchers_per_person),
        "preview": MetadataValue.md(vouchers_per_person.to_markdown()),
    }
    )
    return vouchers_per_person


