import pandas as pd
import statsmodels.formula.api as smf
import pyarrow.parquet as pq
import gc
import time

def reproduce_table2():
    print("=== SD 与 Team Size 的交互效应 ===")
    
    parquet_file = 'OpenAlex_processed.parquet'
    
    columns = [
        'CD Index', 'SD', 'Team Size (log)', # 核心交互变量
        'Title Word Count', 'Title Readability', 'Title Promotional Words (%)',
        'Team Freshness',
        'Career Age', 'Career Age^2', 'Institution H-index', 'Last Author Productivity (log)',
        'year', 'Discipline'
    ]
    
    print(f"正在分块读取数据: {parquet_file} ...")
    try:
        pf = pq.ParquetFile(parquet_file)
        chunks = []
        for batch in pf.iter_batches(batch_size=100000, columns=columns):
            df_chunk = batch.to_pandas().dropna()
            chunks.append(df_chunk)
            print(f"已加载块...", end='\r')
        
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        print(f"\n数据加载完成! 样本量: {len(df)}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {parquet_file}")
        return

    # 运行包含交互项的模型
    print("正在拟合交互效应模型 (可能需要几分钟)...")
    # Q('SD') * Q('Team Size (log)') 会自动包含主效应和交互项
    formula = """
    Q('CD Index') ~ Q('SD') * Q('Team Size (log)') + 
                    Q('Title Word Count') + Q('Title Readability') + Q('Title Promotional Words (%)') +
                    Q('Team Freshness') +
                    Q('Career Age') + I(Q('Career Age')**2) + Q('Institution H-index') + Q('Last Author Productivity (log)') +
                    C(year) + C(Discipline)
    """
    
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())
    
    with open('Result_Table2_Interaction.txt', 'w') as f:
        f.write(model.summary().as_text())
    print("结果已保存至 Result_Table2_Interaction.txt")

if __name__ == "__main__":
    reproduce_table2()
