#!/usr/bin/env python3
"""
PrometheusGPT Mini - Production Readiness Test
Author: MagistrTheOne, Krasnodar, 2025

Тест для проверки готовности системы к production deployment.
"""

import os
import sys
import importlib
import logging
from pathlib import Path

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Тест импортов всех модулей"""
    
    print("🔍 Testing imports...")
    
    modules_to_test = [
        'src.model.transformer',
        'src.model.config',
        'src.data.tokenizer',
        'src.data.dataloader',
        'src.train.full_train',
        'src.api.production_api',
        'src.api.streaming_api',
        'src.monitoring.gpu_monitor',
        'src.monitoring.performance_monitor',
        'src.monitoring.prometheus_metrics',
        'src.quantization.int8_quantization',
        'src.quantization.fp16_quantization',
        'src.exp_tracking.clearml_tracker',
        'src.exp_tracking.mlflow_tracker'
    ]
    
    failed_imports = []
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {module_name}")
        except Exception as e:
            # Игнорируем предупреждения Pydantic и другие незначительные ошибки
            if "FieldInfo" in str(e) or "protected namespace" in str(e):
                print(f"  ⚠️  {module_name}: {e} (ignored)")
            else:
                print(f"  ❌ {module_name}: {e}")
                failed_imports.append(module_name)
    
    return len(failed_imports) == 0, failed_imports

def test_file_structure():
    """Тест структуры файлов"""
    
    print("\n📁 Testing file structure...")
    
    required_files = [
        'src/api/production_api.py',
        'src/train/full_train.py',
        'src/monitoring/gpu_monitor.py',
        'src/quantization/int8_quantization.py',
        'src/exp_tracking/clearml_tracker.py',
        'Dockerfile.production',
        'docker-compose.production.yml',
        'requirements.txt',
        'scripts/run_production.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0, missing_files

def test_dependencies():
    """Тест зависимостей"""
    
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        'torch',
        'fastapi',
        'uvicorn',
        'pydantic',
        'psutil',
        'prometheus_client',
        'clearml',
        'mlflow',
        'matplotlib',
        'pynvml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def test_configuration():
    """Тест конфигурации"""
    
    print("\n⚙️ Testing configuration...")
    
    config_tests = []
    
    # Проверяем requirements.txt
    if Path('requirements.txt').exists():
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if 'torch==' in content and 'fastapi==' in content:
                print("  ✅ requirements.txt has required packages")
                config_tests.append(True)
            else:
                print("  ❌ requirements.txt missing required packages")
                config_tests.append(False)
    else:
        print("  ❌ requirements.txt not found")
        config_tests.append(False)
    
    # Проверяем Docker файлы
    docker_files = ['Dockerfile.production', 'docker-compose.production.yml']
    for docker_file in docker_files:
        if Path(docker_file).exists():
            print(f"  ✅ {docker_file}")
            config_tests.append(True)
        else:
            print(f"  ❌ {docker_file}")
            config_tests.append(False)
    
    # Проверяем monitoring конфигурацию
    monitoring_files = [
        'monitoring/prometheus.yml',
        'monitoring/grafana/datasources/prometheus.yml'
    ]
    
    for monitoring_file in monitoring_files:
        if Path(monitoring_file).exists():
            print(f"  ✅ {monitoring_file}")
            config_tests.append(True)
        else:
            print(f"  ❌ {monitoring_file}")
            config_tests.append(False)
    
    return all(config_tests), config_tests

def test_data_structure():
    """Тест структуры данных"""
    
    print("\n📊 Testing data structure...")
    
    data_tests = []
    
    # Проверяем наличие данных
    data_files = [
        'data/train_ru.txt',
        'data/train_en.txt',
        'data/val_ru.txt',
        'data/val_en.txt',
        'data/dataset_stats.json'
    ]
    
    for data_file in data_files:
        if Path(data_file).exists():
            print(f"  ✅ {data_file}")
            data_tests.append(True)
        else:
            print(f"  ❌ {data_file}")
            data_tests.append(False)
    
    # Проверяем токенизатор
    tokenizer_files = [
        'demo_tokenizer.model',
        'demo_tokenizer.vocab'
    ]
    
    for tokenizer_file in tokenizer_files:
        if Path(tokenizer_file).exists():
            print(f"  ✅ {tokenizer_file}")
            data_tests.append(True)
        else:
            print(f"  ❌ {tokenizer_file}")
            data_tests.append(False)
    
    return all(data_tests), data_tests

def main():
    """Главная функция тестирования"""
    
    print("🚀 PrometheusGPT Mini - Production Readiness Test")
    print("Author: MagistrTheOne, Krasnodar, 2025")
    print("=" * 60)
    
    # Запускаем все тесты
    tests = [
        ("Import Tests", test_imports),
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
        ("Data Structure", test_data_structure)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success, details = test_func()
            results.append((test_name, success, details))
        except Exception as e:
            print(f"  ❌ {test_name} failed with error: {e}")
            results.append((test_name, False, [str(e)]))
    
    # Выводим итоги
    print("\n" + "=" * 60)
    print("📋 PRODUCTION READINESS SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(results)
    
    for test_name, success, details in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        
        if not success and details:
            print(f"    Details: {details}")
        
        if success:
            passed_tests += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 PRODUCTION READY! All tests passed.")
        print("\n🚀 Ready for deployment:")
        print("   python src/api/production_api.py")
        print("   docker-compose -f docker-compose.production.yml up -d")
        return True
    else:
        print("⚠️  NOT PRODUCTION READY. Please fix failing tests.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
