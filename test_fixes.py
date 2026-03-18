#!/usr/bin/env python3
"""
测试脚本：验证编码修复是否有效
"""

import sys
import os

def test_yaml_loading():
    """测试YAML文件加载"""
    try:
        from gl_gym.common.utils import load_env_params, load_model_hyperparams
        print("✅ YAML文件加载测试通过")
        return True
    except Exception as e:
        print(f"❌ YAML文件加载测试失败: {e}")
        return False

def test_csv_loading():
    """测试CSV文件加载"""
    try:
        import pandas as pd
        # 测试一个简单的CSV读取操作
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        test_data.to_csv('test.csv', index=False, encoding='utf-8')
        loaded_data = pd.read_csv('test.csv', encoding='utf-8')
        os.remove('test.csv')
        print("✅ CSV文件加载测试通过")
        return True
    except Exception as e:
        print(f"❌ CSV文件加载测试失败: {e}")
        return False

def test_json_operations():
    """测试JSON文件操作"""
    try:
        import json
        test_data = {'test': 'data'}
        with open('test.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
        with open('test.json', 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        os.remove('test.json')
        print("✅ JSON文件操作测试通过")
        return True
    except Exception as e:
        print(f"❌ JSON文件操作测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试编码修复...")
    print("=" * 50)
    
    tests = [
        test_yaml_loading,
        test_csv_loading,
        test_json_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！编码修复成功。")
        return True
    else:
        print("⚠️  部分测试失败，请检查修复。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

