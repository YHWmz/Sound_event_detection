# 声音事件检测（SED）
# 依赖
```bash
pip install -r requirements.txt
```
# 代码结构
- data：包含了进行数据预处理的脚本以及处理好的feature和label。
- config：实验的配置文件，
- conformer：实验过程中用到的conformer模型
- experiment：实验结果文件夹，./experiments/CDur/best_result下保存了最优配置得到的实验结果。
- dataset.py，losses.py，metrics.py，utils.py：定义了模型训练测试过程中需要用到的函数和类。
- evaluate.py：用于计算指标
- run.py：定义了训练以及测试过程。
- run.sh：运行脚本

# 代码运行
1. 提取特征：

如果在当前路径下运行，则可直接跳过这一步，data文件夹中有已经处理好的数据。
如果不在当前路径下运行，则需删除“./data/dev”，“./data/eval”，“./data/metadata”三个文件夹，
然后运行下面命令对数据重新进行预处理。
```bash
cd data;
bash prepare_data.sh /dssg/home/acct-stu/stu464/data/domestic_sound_events
cd ..;
```


2. 训练、测试：

当前./config/baseline.yaml中的参数即为最佳配置，直接运行下面命令即可
```bash
bash run.sh
```

3. 结果

./experiments/CDur/best_result/下保存了最优配置的实验日志，可以直接运行以下命令计算指标，也可以根据需要修改日志路径。

```bash
python evaluate.py --prediction ./experiments/CDur/best_result/predictions.csv\
                   --label data/eval/label.csv \
                   --output result.txt
```

注 ：即便是固定了所有随机种子，代码运行还是会存在一定随机性，F score可能会有±0.1左右的波动。

