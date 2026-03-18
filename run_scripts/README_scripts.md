# 论文复现实验脚本说明

本目录包含了用于复现GreenLight-Gym2论文实验的所有脚本。

## 脚本列表

### 1. 训练脚本
- **`train_all_models.sh`** / **`train_all_models.bat`**: 训练所有模型
  - 训练PPO、SAC、RecurrentPPO模型
  - 支持确定性和随机性模式
  - 适用于：从头开始训练所有模型

### 2. 评估脚本
- **`run_all_evaluations.bat`**: 完整的评估实验脚本
  - 运行所有确定性、随机性、规则基控制器评估
  - 生成完整的数据文件
  - 适用于：论文复现的核心实验

### 3. 可视化脚本
- **`generate_basic_plots.py`**: 生成可视化图表
  - 轨迹对比图（PPO vs 规则基控制器）
  - 性能指标对比图
  - 不确定性分析图
  - 适用于：生成论文中的图表和分析

### 4. 说明文档
- **`README_scripts.md`**: 详细的使用说明和故障排除指南

## 使用方法

### Windows用户
```bash
# 选项1: 使用现有模型（推荐）
# 1. 运行完整评估实验
run_scripts\run_all_evaluations.bat

# 2. 生成可视化图表
python run_scripts\generate_basic_plots.py

# 选项2: 从头训练模型
# 1. 训练所有模型
run_scripts\train_all_models.bat

# 2. 运行完整评估实验
run_scripts\run_all_evaluations.bat

# 3. 生成可视化图表
python run_scripts\generate_basic_plots.py
```

### Linux/Mac用户
```bash
# 选项1: 使用现有模型（推荐）
# 1. 运行完整评估实验（需要转换为shell脚本）
# 2. 生成可视化图表
python run_scripts/generate_basic_plots.py

# 选项2: 从头训练模型
# 1. 训练所有模型
bash run_scripts/train_all_models.sh

# 2. 运行完整评估实验（需要转换为shell脚本）
# 3. 生成可视化图表
python run_scripts/generate_basic_plots.py
```

## 实验步骤说明

### 1. 环境验证
首先运行环境测试，确保：
- Python环境正常
- 依赖包已安装
- 模型文件存在
- 环境可以正常运行

```bash
python test_env.py --env_id TomatoEnv
```

### 2. 选择实验路径

#### 路径A：使用现有模型（推荐，快速）
如果已有训练好的模型，直接进行评估：

```bash
# 运行完整评估实验
run_scripts\run_all_evaluations.bat

# 生成可视化结果
python run_scripts\generate_basic_plots.py
```

#### 路径B：从头训练模型（完整复现）
如果需要从头训练所有模型：

```bash
# 训练所有模型
run_scripts\train_all_models.bat

# 运行完整评估实验
run_scripts\run_all_evaluations.bat

# 生成可视化结果
python run_scripts\generate_basic_plots.py
```

### 3. 评估实验内容
执行所有必要的评估：
- 确定性PPO评估
- 随机性PPO评估（7个不确定性尺度）
- 规则基控制器评估（确定性和随机性）

### 4. 可视化结果
创建论文中的关键图表：
- 轨迹对比图（PPO vs 规则基控制器）
- 性能指标对比图
- 不确定性分析图

## 输出文件

### 训练数据
- 位置：`train_data/AgriControl/`
- 内容：模型文件、环境统计、训练日志

### 评估结果
- 位置：`data/AgriControl/`
- 内容：CSV格式的评估结果，包含所有状态和性能指标

### 可视化图表
- 位置：当前目录
- 内容：PNG格式的图表文件

## 注意事项

1. **运行时间**：完整实验可能需要数小时，建议在性能较好的机器上运行
2. **存储空间**：确保有足够的磁盘空间存储结果文件
3. **内存要求**：建议至少8GB内存
4. **GPU支持**：脚本默认使用CPU，如需GPU请修改device参数

## 故障排除

### 常见问题
1. **编码错误**：已修复YAML文件读取的编码问题
2. **依赖缺失**：确保已安装所有依赖包
3. **模型文件不存在**：先运行训练脚本或检查模型路径
4. **内存不足**：减少并行环境数量或使用更小的批次大小
5. **VecNormalize文件找不到**：检查模型名称是否正确
   - 确定性模型：`dummy-cg4axdls`
   - 随机性模型：`dummy-nbw883du`
6. **批处理文件中文乱码**：已修复为英文输出
7. **参数错误**：已修复evaluate_baseline.py中的algorithm参数问题

### 调试建议
1. 先运行快速测试脚本验证环境
2. 检查日志文件中的错误信息
3. 确保所有路径和文件名正确
4. 验证Python环境和依赖包版本

## 论文复现要点

本实验复现了论文中的以下关键内容：
- 多算法对比（PPO、SAC、RecurrentPPO）
- 确定性vs随机性性能评估
- 参数不确定性鲁棒性分析
- 规则基控制器基线对比
- 经济性能指标分析
- 约束满足情况评估
