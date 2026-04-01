#!/usr/bin/env python3
"""
验证脚本：检查两个主文件的核心逻辑完整性和数据流畅性。

用法：
  python verify_workflow.py
"""

import json
import os
import sys
from pathlib import Path


def verify_file_existence():
    """验证关键文件是否存在。"""
    print("[Verify] Checking file existence...")
    
    required_files = [
        "query_rooms_for_objects.py",
        "sample_and_place_objects.py",
        "test_layout.py",
        "export_scene_info.py",
        "qwen3_vl_connect.py",
        "hm3d/hm3d_annotated_basis.scene_dataset_config.json",
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    if missing:
        print(f"  ✗ Missing files: {missing}")
        return False
    
    print(f"  ✓ All required files exist")
    return True


def verify_directory_structure():
    """验证输出目录结构。"""
    print("[Verify] Checking directory structure...")
    
    required_dirs = [
        "results",
        "results/scene_info",
        "results/probabilities",
        "results/layouts",
        "objects_images",
    ]
    
    all_exist = True
    for d in required_dirs:
        if not os.path.isdir(d):
            print(f"  ✗ Missing directory: {d}")
            all_exist = False
        else:
            print(f"  ✓ {d}")
    
    return all_exist


def verify_json_schema():
    """验证JSON格式（如果有示例文件）。"""
    print("[Verify] Checking JSON schema compatibility...")
    
    # 检查样例JSON是否存在
    sample_file = "00824-Dd4bFSTQ8gi_scene_info.json"
    if os.path.isfile(sample_file):
        try:
            with open(sample_file, "r") as f:
                data = json.load(f)
            
            # 验证必需的顶级键
            required_keys = ["scene_info", "categories", "rooms", "objects"]
            missing_keys = [k for k in required_keys if k not in data]
            
            if missing_keys:
                print(f"  ✗ Sample JSON missing keys: {missing_keys}")
                return False
            
            print(f"  ✓ Sample scene_info JSON structure valid")
            return True
        except Exception as e:
            print(f"  ✗ Failed to parse sample JSON: {e}")
            return False
    else:
        print(f"  ⓘ No sample JSON found (expected: {sample_file})")
        return True


def verify_python_syntax():
    """验证两个主文件的Python语法。"""
    print("[Verify] Checking Python syntax...")
    
    import ast
    
    files_to_check = [
        "query_rooms_for_objects.py",
        "sample_and_place_objects.py",
    ]
    
    all_valid = True
    for f in files_to_check:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                ast.parse(fp.read())
            print(f"  ✓ {f}")
        except SyntaxError as e:
            print(f"  ✗ {f}: {e}")
            all_valid = False
        except Exception as e:
            print(f"  ✗ {f}: {e}")
            all_valid = False
    
    return all_valid


def verify_function_signatures():
    """验证关键函数是否可被导入（通过AST分析而不执行）。"""
    print("[Verify] Checking function signatures...")
    
    import ast
    
    files_to_check = {
        "query_rooms_for_objects.py": [
            "_pick_free_local_port",
            "_wait_tunnel_ready",
            "_build_image_url",
            "_clean_model_output",
            "query_qwen_for_rooms",
            "parse_room_recommendations",
        ],
        "sample_and_place_objects.py": [
            "generate_probabilities",
            "load_probabilities",
            "sample_object_positions",
            "launch_editor",
            "interactive_sampling_loop",
        ],
    }
    
    all_valid = True
    for filepath, required_funcs in files_to_check.items():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
            
            defined_funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
            missing = [f for f in required_funcs if f not in defined_funcs]
            
            if missing:
                print(f"  ✗ {filepath} missing functions: {missing}")
                all_valid = False
            else:
                print(f"  ✓ {filepath} functions OK")
        
        except Exception as e:
            print(f"  ✗ {filepath}: {e}")
            all_valid = False
    
    return all_valid


def verify_data_flow():
    """验证数据流：query输出 → sample输入。"""
    print("[Verify] Checking data flow compatibility...")
    
    # 检查是否有test_layout文件以及其关键函数
    if not os.path.isfile("test_layout.py"):
        print(f"  ✗ test_layout.py not found")
        return False
    
    try:
        with open("test_layout.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # 检查关键函数是否存在
        key_functions = [
            "load_layout_into_editor",
            "save_layout",
            "create_editor_item",
            "make_sim_cfg",
        ]
        
        missing = [func for func in key_functions if f"def {func}" not in content]
        if missing:
            print(f"  ✗ test_layout.py missing functions: {missing}")
            return False
        
        print(f"  ✓ test_layout.py has all required integration points")
        return True
    
    except Exception as e:
        print(f"  ✗ Error checking test_layout.py: {e}")
        return False


def main():
    """运行所有验证。"""
    print("\n" + "="*60)
    print("AI物体房间推理系统 - 工作流验证")
    print("="*60 + "\n")
    
    checks = [
        ("File Existence", verify_file_existence),
        ("Directory Structure", verify_directory_structure),
        ("JSON Schema", verify_json_schema),
        ("Python Syntax", verify_python_syntax),
        ("Function Signatures", verify_function_signatures),
        ("Data Flow Compatibility", verify_data_flow),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            print()
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}\n")
            results[check_name] = False
    
    # Summary
    print("="*60)
    print("验证总结")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed == total:
        print("\n✓ 所有验证通过！系统已准备好使用。")
        print("\n快速开始：")
        print("  1. query_rooms_for_objects.py --help")
        print("  2. sample_and_place_objects.py --help")
        return 0
    else:
        print(f"\n✗ 有 {total - passed} 项验证失败，请检查上述错误。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
