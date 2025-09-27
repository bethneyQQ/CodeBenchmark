# 测试目录结构说明

## 📁 完整目录结构

```
evaluation_engine/tests/
├── 📋 README.md                    # 详细使用指南
├── ⚡ USAGE.md                     # 快速参考指南  
├── 📂 DIRECTORY_STRUCTURE.md       # 目录结构说明（本文件）
├── 🐍 __init__.py                  # 测试套件索引
│
├── 🔧 core_engines/                # 核心引擎测试
│   ├── 🐍 __init__.py
│   ├── 📋 USAGE.md
│   ├── 📊 test_analysis_engine.py
│   ├── 📏 test_metrics_engine.py
│   ├── 💡 test_prompt_engine.py
│   └── 📈 test_visualization_engine.py
│
├── 🌐 api_integration/             # API和集成测试
│   ├── 🐍 __init__.py
│   ├── 📋 USAGE.md
│   ├── 🔗 test_api_basic.py
│   ├── 🚪 test_api_gateway.py
│   ├── ⚡ test_api_minimal.py
│   ├── 🔄 test_integration.py
│   └── 🔗 test_lm_eval_integration.py
│
├── 🔒 security/                    # 安全和合规测试
│   ├── 🐍 __init__.py
│   ├── 📋 USAGE.md
│   ├── 🛡️ test_security_basic.py
│   └── 🔐 test_security_framework.py
│
├── 📊 analysis_tools/              # 分析工具测试
│   ├── 🐍 __init__.py
│   ├── 📋 USAGE.md
│   ├── 🔍 test_analysis_tools.py
│   ├── 📝 test_analysis_tools_documentation.py
│   ├── 📈 test_analysis_visualization_integration.py
│   └── 🔧 test_fixed_analysis_tools.py
│
└── 🎯 specialized/                 # 专项功能测试
    ├── 🐍 __init__.py
    ├── 📋 USAGE.md
    ├── ⚙️ test_advanced_model_config.py
    ├── 📊 test_composite_metrics.py
    ├── 🔌 test_concrete_model_adapters.py
    ├── 💬 test_single_turn_simple.py
    └── 📋 test_task_2_implementation.py
```

## 📂 目录分类说明

### 🔧 core_engines/ - 核心引擎测试
**用途**: 测试AI评估引擎的核心功能组件
- **分析引擎**: 统计分析、趋势识别、异常检测
- **指标引擎**: NLP指标、代码质量指标、功能指标
- **提示引擎**: 智能提示生成、A/B测试、优化算法
- **可视化引擎**: 图表生成、仪表板、报告导出

### 🌐 api_integration/ - API和集成测试
**用途**: 测试API接口和系统集成功能
- **API网关**: RESTful接口、路由、中间件
- **系统集成**: lm-eval集成、组件互操作
- **基础验证**: 文件结构、依赖检查

### 🔒 security/ - 安全和合规测试
**用途**: 测试安全框架和合规性功能
- **安全框架**: 漏洞扫描、加密管理、访问控制
- **合规管理**: GDPR合规、审计日志、事件检测

### 📊 analysis_tools/ - 分析工具测试
**用途**: 测试数据分析和可视化工具
- **分析工具**: 单轮场景分析、数据处理
- **可视化集成**: 图表生成、交互式可视化
- **文档验证**: API文档、使用示例

### 🎯 specialized/ - 专项功能测试
**用途**: 测试特定功能和高级配置
- **模型配置**: 高级参数配置、动态加载
- **模型适配**: 不同模型的统一接口适配
- **复合指标**: 多维度指标计算和组合
- **特定任务**: 单轮对话、特定实现验证

## 🚀 快速导航

### 按功能运行测试
```bash
# 核心功能测试
pytest evaluation_engine/tests/core_engines/ -v

# API和集成测试
pytest evaluation_engine/tests/api_integration/ -v

# 安全测试
pytest evaluation_engine/tests/security/ -v

# 分析工具测试
pytest evaluation_engine/tests/analysis_tools/ -v

# 专项功能测试
pytest evaluation_engine/tests/specialized/ -v
```

### 按优先级运行测试
```bash
# 高优先级：核心功能 + API集成
pytest evaluation_engine/tests/core_engines/ evaluation_engine/tests/api_integration/ -v

# 中优先级：分析工具 + 专项功能
pytest evaluation_engine/tests/analysis_tools/ evaluation_engine/tests/specialized/ -v

# 低优先级：安全测试（通常最耗时）
pytest evaluation_engine/tests/security/ -v
```

## 📊 测试统计

| 目录 | 测试文件数 | 预估执行时间 | 主要依赖 |
|------|-----------|-------------|----------|
| core_engines | 4 | 30-60秒 | evaluation_engine.core |
| api_integration | 5 | 20-40秒 | lm-eval, fastapi |
| security | 2 | 60-120秒 | cryptography, asyncio |
| analysis_tools | 4 | 15-30秒 | matplotlib, pandas |
| specialized | 5 | 20-45秒 | pyyaml, 模型API |
| **总计** | **20** | **2.5-5分钟** | - |

## 🔧 维护指南

### 添加新测试文件
1. 确定测试文件所属类别
2. 将文件放入相应子目录
3. 更新子目录的USAGE.md
4. 更新主__init__.py中的TEST_DIRECTORIES

### 目录重构
1. 保持现有的5个主要分类
2. 如需新增分类，创建新子目录
3. 更新所有相关文档
4. 确保CI/CD流水线兼容

### 文档维护
- 每个子目录都有独立的USAGE.md
- 主README.md提供整体概览
- USAGE.md提供快速参考
- 本文件提供结构说明

## 🎯 最佳实践

### 测试组织
- 按功能模块分类，而非按文件类型
- 每个子目录功能相对独立
- 保持测试文件命名一致性

### 文档维护
- 及时更新使用说明
- 保持示例代码的有效性
- 记录常见问题和解决方案

### 执行策略
- 优先运行核心功能测试
- 并行执行独立的测试类别
- 定期运行完整测试套件