#!/usr/bin/env python3
"""
性能比较脚本：对比 CPU 和 GPU 优化器的迭代步数和速度
"""
import time
import math

from sources.data import Case
from sources.data.constraint import VertexVertexDistance
from sources.data.topo import Vertex, Solid
from sources.geometry.Point import Point
from sources.geometry.Cartesian import Cartesian
from sources.operation import distance, transform
from sources.solver.Darwin.optimizer.optimizer import DarwinOptimizer
from sources.solver.Darwin.evaluator import VertexVertexDistanceEvaluator


def create_test_case():
    """创建与测试用例相同的问题"""
    case = Case('PerformanceTest', 'CPU vs GPU performance comparison')

    SQRT2 = 2 ** 0.5

    origin_a = Point(SQRT2, 0, 0)
    x_axis_point_a = Point(1.5 * SQRT2, 0, 0.5 * SQRT2)
    z_axis_point_a = Point(0.5 * SQRT2, 0, 0.5 * SQRT2)
    cartesian_a = Cartesian(origin_a, x_axis_point_a, z_axis_point_a)
    solid_a = Solid(cartesian_a)
    vertex_a = Vertex()
    solid_a.add_vertex(vertex_a)
    case.add_solid(solid_a)

    origin_b = Point(0, SQRT2, 0)
    x_axis_point_b = Point(1, SQRT2, 0)
    z_axis_point_b = Point(0, 0.5 * SQRT2, 0.5 * SQRT2)
    cartesian_b = Cartesian(origin_b, x_axis_point_b, z_axis_point_b)
    solid_b = Solid(cartesian_b)
    vertex_b = Vertex()
    solid_b.add_vertex(vertex_b)
    case.add_solid(solid_b)

    vertices_distance_constraint = VertexVertexDistance(vertex_a, vertex_b, 5.0)
    case.add_constraint(vertices_distance_constraint)

    return case, vertex_a, vertex_b


def run_optimization(device: str) -> tuple:
    """运行优化并返回 (最终score, 迭代步数, 耗时秒)"""
    case, vertex_a, vertex_b = create_test_case()
    
    # 创建评估器
    constraint = case.constraints[0]
    evaluator = VertexVertexDistanceEvaluator(constraint, case.solids)
    
    # 创建优化器，强制指定设备
    optimizer = DarwinOptimizer(device=device)
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行优化
    final_score, iterations = optimizer.optimize(
        evaluator=evaluator,
        solids=case.solids,
        vertex1=vertex_a,
        vertex2=vertex_b
    )
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return final_score, iterations, elapsed_time


def main():
    print("=" * 60)
    print("CPU vs GPU 性能比较")
    print("=" * 60)
    
    results = {}
    
    # 测试 CPU 模式
    print("\n" + "=" * 60)
    print("测试 CPU 模式...")
    print("=" * 60)
    cpu_score, cpu_iterations, cpu_time = run_optimization(device='cpu')
    results['cpu'] = (cpu_score, cpu_iterations, cpu_time)
    
    print(f"\nCPU 结果:")
    print(f"  最终 score: {cpu_score:.2e}")
    print(f"  迭代步数: {cpu_iterations}")
    print(f"  耗时: {cpu_time:.4f} 秒")
    
    # 测试 GPU 模式
    print("\n" + "=" * 60)
    print("测试 GPU 模式...")
    print("=" * 60)
    gpu_score, gpu_iterations, gpu_time = run_optimization(device='cuda')
    results['gpu'] = (gpu_score, gpu_iterations, gpu_time)
    
    print(f"\nGPU 结果:")
    print(f"  最终 score: {gpu_score:.2e}")
    print(f"  迭代步数: {gpu_iterations}")
    print(f"  耗时: {gpu_time:.4f} 秒")
    
    # 比较结果
    print("\n" + "=" * 60)
    print("性能比较总结")
    print("=" * 60)
    print(f"{'指标':<20} {'CPU':<20} {'GPU':<20}")
    print("-" * 60)
    print(f"{'最终 score':<20} {results['cpu'][0]:<20.2e} {results['gpu'][0]:<20.2e}")
    print(f"{'迭代步数':<20} {results['cpu'][1]:<20} {results['gpu'][1]:<20}")
    print(f"{'耗时 (秒)':<20} {results['cpu'][2]:<20.4f} {results['gpu'][2]:<20.4f}")
    
    # 计算速度比
    if results['gpu'][2] > 0:
        speedup = results['cpu'][2] / results['gpu'][2]
        print(f"\nGPU 速度提升: {speedup:.2f}x")
        
        if speedup > 1:
            print("结论: GPU 更快")
        else:
            print("结论: CPU 更快")
    
    # 检查收敛
    print("\n" + "=" * 60)
    print("收敛情况")
    print("=" * 60)
    for mode, (score, iterations, elapsed) in results.items():
        if score < 1e-6:
            print(f"{mode.upper()}: ✓ 已收敛 (score < 1e-6)")
        else:
            print(f"{mode.upper()}: ✗ 未收敛 (score = {score:.2e})")


if __name__ == '__main__':
    main()
