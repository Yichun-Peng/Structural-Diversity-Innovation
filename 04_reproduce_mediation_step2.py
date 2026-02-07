import pandas as pd
import statsmodels.formula.api as smf
import pyarrow.parquet as pq
import gc

def reproduce_mediation_step2():
    print("=== (Step 2): SD + DI -> CD Index (Path b & c') ===")
    
    parquet_file = 'OpenAlex_Full_Data.parquet'

    columns = [
        'CD Index', 'SD', 'DI',
        'Title Word Count', 'Title Readability', 'Title Promotional Words (%)',
        'Team Size (log)', 'Team Freshness',
        'Career Age', 'Career Age^2', 'Institution H-index', 'Last Author Productivity (log)',
        'year', 'Discipline'
    ]
    
    print(f"正在读取数据 (Step 2)...")
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

    print(f"\n样本量: {len(df)}. 正在拟合 Path b & c' 模型...")
    
    # 路径 B 和 C': 将中介变量 DI 加入回归
    formula = """
    Q('CD Index') ~ Q('SD') + Q('DI') +
                    Q('Title Word Count') + Q('Title Readability') + Q('Title Promotional Words (%)') +
                    Q('Team Size (log)') + Q('Team Freshness') +
                    Q('Career Age') + I(Q('Career Age')**2) + Q('Institution H-index') + Q('Last Author Productivity (log)') +
                    C(year) + C(Discipline)
    """
    
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())
    
    with open('Result_Mediation_Step2.txt', 'w') as f:
        f.write(model.summary().as_text())

    # 计算间接效应所需信息
    beta_di = model.params["Q('DI')"]
    beta_sd_direct = model.params["Q('SD')"]
    
    print("-" * 30)
    print(f">>> Path b (DI -> CD): {beta_di:.4f}")
    print(f">>> Path c' (SD -> CD, Direct): {beta_sd_direct:.4f}")
    print("-" * 30)
    print("提示: 间接效应 (Indirect Effect) = (Step1中的SD系数) * (Step2中的DI系数)")

if __name__ == "__main__":
    reproduce_mediation_step2()
