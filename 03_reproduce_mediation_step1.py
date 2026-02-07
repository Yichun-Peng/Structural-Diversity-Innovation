import pandas as pd
import statsmodels.formula.api as smf
import pyarrow.parquet as pq
import gc

def reproduce_mediation_step1():
    print("=== (Step 1): SD -> DI (Path a) ===")
    
    parquet_file = 'OpenAlex_Full_Data.parquet'

    columns = [
        'DI', 'SD', 
        'Title Word Count', 'Title Readability', 'Title Promotional Words (%)',
        'Team Size (log)', 'Team Freshness',
        'Career Age', 'Career Age^2', 'Institution H-index', 'Last Author Productivity (log)',
        'year', 'Discipline'
    ]
    
    print(f"正在读取数据 (Step 1)...")
    try:
        pf = pq.ParquetFile(parquet_file)
        chunks = []
        for batch in pf.iter_batches(batch_size=100000, columns=columns):
            chunks.append(batch.to_pandas().dropna())
            print(f"Loading...", end='\r')
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
    except Exception as e:
        print(f"读取错误: {e}")
        return

    print(f"\n样本量: {len(df)}. 正在拟合 Path a 模型...")
    
    formula = """
    Q('DI') ~ Q('SD') + 
              Q('Title Word Count') + Q('Title Readability') + Q('Title Promotional Words (%)') +
              Q('Team Size (log)') + Q('Team Freshness') +
              Q('Career Age') + I(Q('Career Age')**2) + Q('Institution H-index') + Q('Last Author Productivity (log)') +
              C(year) + C(Discipline)
    """
    
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())
    
    with open('Result_Mediation_Step1.txt', 'w') as f:
        f.write(model.summary().as_text())
        
    print(f"\n>>> 关键系数 (Path a): SD -> DI = {model.params["Q('SD')"]:.4f}")

if __name__ == "__main__":
    reproduce_mediation_step1()
