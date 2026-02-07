import pandas as pd
import statsmodels.formula.api as smf
import pyarrow.parquet as pq
import gc
import time

def reproduce_table1():
    print("=== SD 对 CD Index 的主效应 ===")

    parquet_file = 'OpenAlex_processed.parquet'

    columns = [
        'CD Index', 'SD', 
        'Title Word Count', 'Title Readability', 'Title Promotional Words (%)',
        'Team Size (log)', 'Team Freshness',
        'Career Age', 'Career Age^2', 'Institution H-index', 'Last Author Productivity (log)',
        'year', 'Discipline'
    ]
    
    print(f"正在分块读取数据: {parquet_file} ...")
    start_time = time.time()
    
    try:
        pf = pq.ParquetFile(parquet_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {parquet_file}。请修改代码中的文件路径。")
        return

    chunks = []
    # 分块读取，每块 10 万行
    for batch in pf.iter_batches(batch_size=100000, columns=columns):

        df_chunk = batch.to_pandas()
        df_chunk.dropna(inplace=True)
        
        chunks.append(df_chunk)
        print(f"已加载数据块... (当前累计行数预估: {len(chunks)*100000})", end='\r')
    
    print("\n正在合并所有数据块...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    print(f"数据加载完成! 最终样本量: {len(df)}")
    print(f"耗时: {time.time() - start_time:.2f} 秒")

    # 运行 OLS 回归
    print("正在拟合 OLS 模型...")
    # 使用 Q("name") 处理包含空格的变量名
    formula = """
    Q('CD Index') ~ Q('SD') + 
                    Q('Title Word Count') + Q('Title Readability') + Q('Title Promotional Words (%)') +
                    Q('Team Size (log)') + Q('Team Freshness') +
                    Q('Career Age') + I(Q('Career Age')**2) + Q('Institution H-index') + Q('Last Author Productivity (log)') +
                    C(year) + C(Discipline)
    """
    
    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())

    with open('Result_Table1_Main_Effect.txt', 'w') as f:
        f.write(model.summary().as_text())
    print("结果已保存至 Result_Table1_Main_Effect.txt")

if __name__ == "__main__":
    reproduce_table1()
