import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import argparse
import copy
import pickle
from datetime import datetime, timezone
from pathlib import Path

# 导入部装setup矩阵
from product_information.setup_matrix import Part_matrix
# 导入总装setup矩阵
from product_information.setup_matrix import Assembly_matrix
# 导入产品信息
from multiprocessing import Pool, cpu_count
import itertools
import random
from purchase_mutation import purchase_mutate_chromosome, mutate_assembly_orders,swap_machine_selection,mutate_with_machine_remap_positions,mutate_machine_selection_multi_dynamic, mutate_shuffle_positions_dual, mutate_final_assembly_machine_selection

from plt_history import plot_pareto_trend_subplots
from evolution_logger import EvolutionLogger
from prompt_builder import build_prompt
# from Addition.similarity import hamming_similarity, flatten, structure_similarity, find_top_k_similar_max_or_avg
from Addition.similarity import find_top_k_similar_max_or_avg

from mutil_agent_generation_individual.prompt_builder3 import build_prompt_for_llm, get_purchase_ranges, build_assembly_count_list
# from Addition.config import ordersss,order_delay_costss,product_Warehousing_Costss,order_Deadliness,ELITE_RATIO,RANDOM_GA_RATIO,LLM_GUIDED_RATIO
from config3 import ordersss,order_delay_costss,product_Warehousing_Costss,order_Deadliness,ELITE_RATIO,RANDOM_GA_RATIO,LLM_GUIDED_RATIO,Product_library
# print(LLM_GUIDED_RATIO)

from mutil_agent_generation_individual.llm_agent import individualAgent
from mutil_agent_generation_individual.DeepSeek import llm_configuration
from mutil_agent_generation_individual.reproducibility import configure_reproducibility, finalize_reproducibility
from pymoo.indicators.hv import HV


# Python 内置 random
DEFAULT_ALGORITHM_SEED = 4
random.seed(DEFAULT_ALGORITHM_SEED)

# NumPy
np.random.seed(DEFAULT_ALGORITHM_SEED)


# 设置中文字体，解决乱码问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体，你也可以选择其他合适的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def get_product_type(demands, product_id):
    cumulative_sum = 0
    for i, demand in enumerate(demands):
        cumulative_sum += demand
        if product_id < cumulative_sum:
            return i
    return None

def get_subassembly_operations(product_info):
    """
    从产品工序信息中筛选出部装工序（工序类型为 1）
    """
    return [op for op in product_info if op['工序类型'] == 1]


def is_valid_sequence(sequence, operations):
    """
    检查一个工序顺序是否合法，即前置工序总是在后续工序之前执行
    """
    index_map = {seq: i for i, seq in enumerate(sequence)}
    for op in operations:
        for pred in op['前置工序']:
            if pred in index_map and index_map[pred] > index_map[op['工序编号']]:
                return False
    return True


def generate_valid_subassembly_sequences(product_info):
    """
    生成一个产品的所有合法部装工序顺序
    """
    # # print(
    #     "这是product_info",product_info
    # )
    subassembly_ops = get_subassembly_operations(product_info)
    # # print("这是subassembly_ops",subassembly_ops)
    all_permutations = itertools.permutations([op['工序编号'] for op in subassembly_ops])
    # # print("这是all_permutations",all_permutations)
    valid_sequences = []
    for perm in all_permutations:
        if is_valid_sequence(perm, subassembly_ops):
            valid_sequences.append(list(perm))
    return valid_sequences


def determine_internal_sequences(products_info, demands):
    all_sequences = []
    product_count = 0
    for i, product_info in enumerate(products_info):
        valid_sequences = generate_valid_subassembly_sequences(product_info)

        # # print("这是valid_sequences",valid_sequences)
        # input("暂停")
        for _ in range(demands[i]):
            if valid_sequences:
                # 随机选择一个合法的顺序
                chosen_sequence = random.choice(valid_sequences)
            else:
                chosen_sequence = []
            all_sequences.append((product_count, chosen_sequence))
            product_count += 1
    return all_sequences

def generate_individual(orders, product_library):

    individual1 = []
    for product_type, order in enumerate(orders):
        product_index = product_type
        product = product_library[product_index]
        # # print("这是产品1",product)
        quantity = order
        for _ in range(quantity):
            for process in product:
                if process['工序类型'] == 0:
                    num_suppliers = len(process['供应商'])
                    individual1.append(random.randint(0, num_suppliers - 1))

    # 确定全部产品内部的部装加工顺序
    internal_sequences = determine_internal_sequences(product_library, orders)

    # # print("这是internal_sequences",internal_sequences)
    # input("暂停")

    # 生成工序的顺序列表
    process_sequence_list = []
    product_internal_sequences = []

    for product_index, internal_seq in internal_sequences:
        product_internal_sequences.append(internal_seq)
        for operation in internal_seq:
            process_sequence_list.append(product_index)

    # 打乱工序顺序列表
    random.shuffle(process_sequence_list)



    # 输出产品内部的部装工序顺序
    # # print("产品内部的部装工序顺序：")
    # for idx, seq in enumerate(product_internal_sequences, start=1):
    #     # print(f"产品 {idx}: {seq}")

    # 输出仅含产品编号的工序顺序列表
    # # print("\n仅含产品编号的工序顺序列表：")
    # # print(process_sequence_list)

    # 为每个工序随机分配可用机器
    machine_choice_list = []
    num_count = {}
    for index, product_index in enumerate(process_sequence_list):
        product_type = get_product_type(orders, product_index)
        num_count[product_index] = num_count.get(product_index, 0) + 1
        if product_type is not None:
            # print(f"这是对应的产品编号 {product_index}，它是 process_sequence_list 中的第 {index + 1} 个，是数字 {product_index} 的第 {num_count[product_index]} 次出现，属于类型 {product_type}")
            pass
        else:
            print(f"编号 {product_index} 超出了产品编号范围。")

        product_goxu = int(product_internal_sequences[product_index][num_count[product_index] - 1]) - 1
        # # print("这是这个product_index对应产品的工序编号：", product_goxu)
        #
        # # print("这是这个工序对应的信息")
        # # print(product_library[product_type][product_goxu])
        # # print("这是这个工序对应的机器：", product_library[product_type][product_goxu]['加工机器'])
        # # print("这是这个工序选择的机器编号")
        random_index = random.randint(0, len(product_library[product_type][product_goxu]['加工机器']) - 1)
        # # print(random_index)
        machine_choice_list.append(random_index)

    # 输出机器选择列表
    # # print("\n机器选择列表：")
    # # print(machine_choice_list)

    # 生成总装工序的顺序列表

    # 生成 0 到 n 的产品编号列表
    final_process_sequence_list = list(range(sum(orders)))

    # 随机打乱产品列表以得到总装顺序
    random.shuffle(final_process_sequence_list)

    # # print("随机生成的总装顺序:", final_process_sequence_list)

    final_machine_choice_list = []

    for index, product_index in enumerate(final_process_sequence_list):
        product_type = get_product_type(orders, product_index)
        # # print("这是这个产品总装对应的机器选择：", product_library[product_type][-1]['加工机器'])

        random_index = random.randint(0, len(product_library[product_type][-1]['加工机器']) - 1)
        final_machine_choice_list.append(random_index)

    # # print("这是总装的机器选择列表：",final_machine_choice_list)

    individual = {
        "采购供应商选择": individual1,
        "产品内部的部装工序顺序": product_internal_sequences,
        "部装工序顺序列表": process_sequence_list,
        "部装机器选择列表": machine_choice_list,
        "总装工序顺序列表": final_process_sequence_list,
        "总装机器选择列表": final_machine_choice_list
    }

    # print(individual)



    return individual

def generate_population(population_size, orders, product_library):
    return [generate_individual(orders, product_library) for _ in range(population_size)]


def computer_caigou_cost(caigou_liat, product_library):
    total_cost = 0

    all_purchase_costs = []
    for product in product_library:
        purchase_costs = []
        for process in product:
            if process['工序类型'] == 0:
                purchase_costs.append(process['单位成本'])
        all_purchase_costs.append(purchase_costs)

    # # print("all_purchase_costs",all_purchase_costs)

    for product_index, product_purchases in enumerate(caigou_liat):
        product_costs = all_purchase_costs[product_index]
        for purchase in product_purchases:
            for process_index, supplier_index in enumerate(purchase):
                cost = product_costs[process_index][supplier_index]
                total_cost += cost

    # # print("total_cost",total_cost)

    return total_cost

def simulate_purchase_time_fluctuation(product_library, caigou_liat):
    fluctuated_times = []
    for product_index, product_purchases in enumerate(caigou_liat):
        product = product_library[product_index]
        purchase_processes = [p for p in product if p['工序类型'] == 0]
        product_fluctuated_times = []
        for purchase in product_purchases:
            purchase_fluctuated_times = []
            for process_index, supplier_index in enumerate(purchase):
                process = purchase_processes[process_index]
                base_time = process['采购时间'][supplier_index]
                # 生成纯波动时间，范围在正负10%之间
                fluctuation = random.uniform(-base_time * 0.1, base_time * 0.1)
                # fluctuation = random.uniform(0,0)
                purchase_fluctuated_times.append(fluctuation)
            product_fluctuated_times.append(purchase_fluctuated_times)
        fluctuated_times.append(product_fluctuated_times)
    return fluctuated_times

# def find_and_modify_min_values(lst, positions):
#     min_values = []
#     # 分别找出每个指定位置的最小值
#     for pos in positions:
#         current_min = min([sublist[pos] for sublist in lst])
#         min_values.append(current_min)
#     # 找出这些最小值中的最大值
#     max_min = max(min_values)
#     # 将所有最小值所在位置的值替换为无穷大
#     for pos, min_val in zip(positions, min_values):
#         for sublist in lst:
#             if sublist[pos] == min_val:
#                 sublist[pos] = float('inf')
#     return lst, max_min

def find_and_modify_min_values(lst, positions):
    import copy
    lst = copy.deepcopy(lst)  # 避免修改原始数据
    min_values = []
    min_positions = []

    for pos in positions:
        non_zero_values = [(i, sublist[pos]) for i, sublist in enumerate(lst) if sublist[pos] != 0 and sublist[pos] != float('inf')]
        if not non_zero_values:
            return "错误：指定位置的最小值只有 0 或 inf", None
        i_min, current_min = min(non_zero_values, key=lambda x: x[1])
        min_values.append(current_min)
        min_positions.append((i_min, pos))

    max_min = max(min_values)

    # 精确替换每个最小值的位置
    for i, pos in min_positions:
        lst[i][pos] = float('inf')

    return lst, max_min

# def find_and_modify_min_values2(lst, positions):
#
#     # print("函数里面的输入是", lst)
#     # print("函数里面的positions是", positions)
#     min_values = []
#     # 分别找出每个指定位置除 0 之外的最小值
#     for pos in positions:
#         non_zero_values = [sublist[pos] for sublist in lst if sublist[pos] != 0]
#         if not non_zero_values:
#             return "错误：指定位置的最小值只有 0", None
#         current_min = min(non_zero_values)
#         min_values.append(current_min)
#     # 找出这些最小值中的最大值
#     max_min = max(min_values)
#     # 将所有最小值所在位置的值替换为无穷大
#     for pos, min_val in zip(positions, min_values):
#         for sublist in lst:
#             if sublist[pos] == min_val:
#                 sublist[pos] = float('inf')
#
#     # print("函数里面的输出是", lst)
#     # print("函数里面的max_min是", max_min)
#     return lst, max_min

def find_and_modify_min_values2(lst, positions):
    import copy
    lst = copy.deepcopy(lst)  # 避免修改原始数据
    min_values = []
    min_positions = []

    for pos in positions:
        non_zero_values = [(i, sublist[pos]) for i, sublist in enumerate(lst) if sublist[pos] != 0 and sublist[pos] != float('inf')]
        if not non_zero_values:
            return "错误：指定位置的最小值只有 0 或 inf", None
        i_min, current_min = min(non_zero_values, key=lambda x: x[1])
        min_values.append(current_min)
        min_positions.append((i_min, pos))

    max_min = max(min_values)

    # 精确替换每个最小值的位置
    for i, pos in min_positions:
        lst[i][pos] = float('inf')

    return lst, max_min

def find_product_type(orderss, product_number):
    # # print("这是在find_product_type里面",orderss)
    # # print("这是在find_product_type里面",product_number)
    index = 0
    cumulative_count = 0
    for i, order in enumerate(orderss):
        cumulative_count += order
        if product_number < cumulative_count:
            return i
    return None

def get_occurrence_at_position(lst, position):
    if position < 0 or position >= len(lst):
        return None
    target_num = lst[position]
    return lst[:position + 1].count(target_num)

def find_product_position(orders, product_index):
    start = 0
    for index, num in enumerate(orders):
        end = start + num
        if start <= product_index < end:
            return product_index - start
        start = end
    return None

def find_rank(numbers, target):
    try:
        # 把列表里的字符串元素转为整数
        int_numbers = [int(num) for num in numbers]
        # 对转换后的整数列表进行排序
        sorted_numbers = sorted(int_numbers)
        # 找到目标数字在排序后列表中的索引
        return sorted_numbers.index(int(target))
    except ValueError:
        # 如果目标数字不在列表中，返回 None
        return None
    except Exception:
        # 处理其他可能的异常
        return None

def merge_orders(orders):
    """
    合并多个订单，计算每种产品类型的总采购量

    参数:
        orders: 订单列表，每个元素是一个列表，表示单个订单中各产品类型的采购量

    返回:
        total_order: 合并后的总订单，列表元素为各产品类型的总采购量
    """
    if not orders:  # 处理空订单列表情况
        return []

    # 确定产品类型数量（假设所有订单结构一致）
    product_type_count = len(orders[0])

    # 初始化总订单，所有产品类型初始采购量为0
    total_order = [0] * product_type_count

    # 遍历每个订单，累加各产品类型的采购量
    for order in orders:
        # 检查当前订单的产品类型数量是否与第一个订单一致
        if len(order) != product_type_count:
            raise ValueError("所有订单必须包含相同数量的产品类型")

        for i in range(product_type_count):
            # 确保订单中的数量是非负整数
            if not isinstance(order[i], int) or order[i] < 0:
                raise ValueError("订单中的产品数量必须是非负整数")

            total_order[i] += order[i]

    return total_order

def generate_product_mappings(orders):
    """
    生成产品编号对应的所属订单和产品类型列表

    参数:
        orders: 订单列表，每个子列表表示订单中各产品类型的采购量

    返回:
        order_list: 索引为产品编号，值为所属订单索引
        type_list: 索引为产品编号，值为产品类型
    """
    order_list = []
    type_list = []
    current_product_id = 0

    # 按产品类型(0→1→2)和订单顺序遍历
    for product_type in range(len(orders[0])):  # 自动适配产品类型数量
        for order_idx, order in enumerate(orders):
            count = order[product_type]
            # 为每个产品添加信息到列表
            for _ in range(count):
                order_list.append(order_idx)  # 订单索引（0开始）
                type_list.append(product_type)
                current_product_id += 1

    return order_list, type_list

def calculate_total_costs(process_times, order_list, type_list,
                          order_delay_cost, order_deadline, product_warehousing_cost):
    """
    计算总延迟成本和总仓储成本

    参数:
        process_times: 产品加工时间列表
        order_list: 产品所属订单索引列表
        type_list: 产品类型索引列表
        order_delay_cost: 订单单位延迟成本
        order_deadline: 订单截止时间
        product_warehousing_cost: 产品类型单位仓储成本

    返回:
        total_delay: 总延迟成本
        total_warehousing: 总仓储成本
    """
    # print("process_times",process_times)
    # print("len(process_times)",len(process_times))
    # 输出process_times的每个元素
    # for i, process_time in enumerate(process_times):
    #     print(f"process_times[{i}] = {process_time}")
    # print("order_list",order_list)
    # print("type_list",type_list)
    # print("order_delay_cost",order_delay_cost)
    # print("order_deadline",order_deadline)
    # print("product_warehousing_cost",product_warehousing_cost)
    # input("暂停")
    # 1. 计算每个订单的完成时间
    order_completion = {}
    for product_id in range(len(order_list)):
        order_idx = order_list[product_id]
        product_end = process_times[product_id][1]
        if order_idx not in order_completion or product_end > order_completion[order_idx]:
            order_completion[order_idx] = product_end

    # 2. 计算总延迟成本
    total_delay = 0.0
    for order_idx, completion_time in order_completion.items():
        if completion_time > order_deadline[order_idx]:
            delay = completion_time - order_deadline[order_idx]
            total_delay += delay * order_delay_cost[order_idx]

    # 3. 计算总仓储成本
    total_warehousing = 0.0
    for product_id in range(len(order_list)):
        order_idx = order_list[product_id]
        product_type = type_list[product_id]
        storage_time = order_completion[order_idx] - process_times[product_id][1]
        if storage_time > 0:
            total_warehousing += storage_time * product_warehousing_cost[product_type]

    return total_delay, total_warehousing


# def dynamic_select_next_generation(population, multi_obj_values, elite_ratio):
#     """
#     按帕累托前沿筛选下一代个体（不裁剪）：
#     1. 提取当前种群的帕累托前沿
#     2. 全部保留作为精英个体，不限制数量
#
#     参数:
#         population: 当前种群（任意大小）
#         multi_obj_values: 所有个体的多目标值（[[c1,c2,c3], ...]）
#         elite_ratio: 保留比例参数（此版本不使用该参数）
#
#     返回:
#         next_pop: 下一代种群（全部帕累托前沿个体）
#         selected_indices: 选中个体的原始索引
#         selected_values: 选中个体的多目标值
#     """
#     pop_size = len(population)
#     if pop_size == 0:
#         return [], [], []
#
#     # --------------------------
#     # 步骤1：提取帕累托前沿个体
#     # --------------------------
#     def is_dominated(p, others):
#         p = np.array(p)
#         others = np.array(others)
#         return np.any(np.all(others <= p, axis=1) & np.any(others < p, axis=1))
#
#     pareto_indices = []
#     for i, p in enumerate(multi_obj_values):
#         if not is_dominated(p, multi_obj_values):
#             pareto_indices.append(i)
#
#     # --------------------------
#     # 步骤2：提取结果（不裁剪）
#     # --------------------------
#     selected_indices = pareto_indices
#     next_pop = [population[idx] for idx in selected_indices]
#     selected_values = [multi_obj_values[idx] for idx in selected_indices]
#
#     return next_pop, selected_indices, selected_values

# def dynamic_select_next_generation(population, multi_obj_values, elite_ratio):
#     """
#     从当前帕累托前沿中选择精英个体（基于 HV 贡献裁剪，并强制保留总成本最小解）
#
#     参数:
#         population: 当前种群（list）
#         multi_obj_values: 所有个体的多目标值（list of [c1, c2, c3] 或 ndarray）
#         elite_ratio: 精英比例（如 0.1 表示保留 10%）
#
#     返回:
#         next_pop: 下一代种群（精英个体）
#         selected_indices: 选中个体的原始索引
#         selected_values: 选中个体的多目标值
#         min_cost_solution: 总成本最小的解（个体, 值, 索引）
#     """
#     pop_size = len(population)
#     if pop_size == 0:
#         return [], [], [], None
#
#     # --------------------------
#     # 输入统一转为 numpy 数组
#     # --------------------------
#     multi_obj_values = np.array(multi_obj_values, dtype=float)
#
#     # --------------------------
#     # 步骤1：提取帕累托前沿个体
#     # --------------------------
#     def is_dominated(p, others):
#         return np.any(np.all(others <= p, axis=1) & np.any(others < p, axis=1))
#
#     pareto_indices = []
#     for i, p in enumerate(multi_obj_values):
#         if not is_dominated(p, multi_obj_values):
#             pareto_indices.append(i)
#
#     pareto_values = multi_obj_values[pareto_indices]
#     pareto_values = np.unique(pareto_values, axis=0)
#
#     # --------------------------
#     # 步骤2：限制精英数量 (减1，为总成本最小解留位置)
#     # --------------------------
#     max_elite_size = max(1, round(pop_size * elite_ratio) - 1)
#
#     if len(pareto_values) <= max_elite_size:
#         selected_values = pareto_values.tolist()
#     else:
#         # HV 贡献裁剪
#         reference_point = np.max(multi_obj_values, axis=0) * 1.2
#         hv_calculator = HV(ref_point=reference_point)
#
#         total_hv = hv_calculator.do(pareto_values)
#         contributions = []
#         for i in range(len(pareto_values)):
#             subset = np.delete(pareto_values, i, axis=0)
#             hv_without_i = hv_calculator.do(subset)
#             contribution = total_hv - hv_without_i
#             contributions.append((i, contribution))
#
#         top_indices = sorted(contributions, key=lambda x: -x[1])[:max_elite_size]
#         selected_values = [pareto_values[i].tolist() for i, _ in top_indices]
#
#     # --------------------------
#     # 步骤3：强制保留总成本最小解
#     # --------------------------
#     total_costs = np.sum(multi_obj_values, axis=1)
#     min_cost_idx = int(np.argmin(total_costs))
#     min_cost_val = multi_obj_values[min_cost_idx].tolist()
#
#     if not any(np.allclose(min_cost_val, sv) for sv in selected_values):
#         selected_values.append(min_cost_val)
#
#     # --------------------------
#     # 步骤4：回溯原始索引和个体
#     # --------------------------
#     selected_indices = []
#     next_pop = []
#     for val in selected_values:
#         for i, v in enumerate(multi_obj_values.tolist()):
#             if np.allclose(v, val) and i not in selected_indices:
#                 selected_indices.append(i)
#                 next_pop.append(population[i])
#                 break
#
#     selected_values = [multi_obj_values[idx].tolist() for idx in selected_indices]
#     # min_cost_solution = (population[min_cost_idx], min_cost_val, min_cost_idx)
#
#     # return next_pop, selected_indices, selected_values, min_cost_solution
#     return next_pop, selected_indices, selected_values
def dynamic_select_next_generation(population, multi_obj_values, elite_ratio):
    """
    精英选择：一半来自帕累托前沿的 HV 贡献最大个体，一半来自剩余种群中总成本最小个体

    参数:
        population: 当前种群（list）
        multi_obj_values: 所有个体的多目标值（list of [c1, c2, c3] 或 ndarray）
        elite_ratio: 精英比例（如 0.1 表示保留 10%）

    返回:
        next_pop: 下一代种群（精英个体）
        selected_indices: 选中个体的原始索引
        selected_values: 选中个体的多目标值
    """
    pop_size = len(population)
    if pop_size == 0:
        return [], [], []

    multi_obj_values = np.array(multi_obj_values, dtype=float)
    elite_count = max(2, round(pop_size * elite_ratio))  # 至少保留两个
    half_elite = elite_count // 2

    # --------------------------
    # 步骤1：提取帕累托前沿个体
    # --------------------------
    def is_dominated(p, others):
        return np.any(np.all(others <= p, axis=1) & np.any(others < p, axis=1))

    pareto_indices = [i for i, p in enumerate(multi_obj_values) if not is_dominated(p, multi_obj_values)]
    pareto_values = multi_obj_values[pareto_indices]

    # --------------------------
    # 步骤2：从帕累托前沿中选 HV 贡献最大的 half_elite 个体
    # --------------------------
    reference_point = np.max(multi_obj_values, axis=0) * 1.2
    hv_calculator = HV(ref_point=reference_point)

    total_hv = hv_calculator.do(pareto_values)
    contributions = []
    for i in range(len(pareto_values)):
        subset = np.delete(pareto_values, i, axis=0)
        hv_without_i = hv_calculator.do(subset)
        contribution = total_hv - hv_without_i
        contributions.append((i, contribution))

    top_indices = sorted(contributions, key=lambda x: -x[1])[:half_elite]
    hv_selected_indices = [pareto_indices[i] for i, _ in top_indices]

    # --------------------------
    # 步骤3：从剩余个体中选总成本最小的 half_elite 个体
    # --------------------------
    remaining_indices = list(set(range(pop_size)) - set(hv_selected_indices))
    remaining_costs = np.sum(multi_obj_values[remaining_indices], axis=1)
    cost_sorted = np.argsort(remaining_costs)[:half_elite]
    cost_selected_indices = [remaining_indices[i] for i in cost_sorted]

    # --------------------------
    # 步骤4：合并精英个体
    # --------------------------
    selected_indices = hv_selected_indices + cost_selected_indices
    next_pop = [population[i] for i in selected_indices]
    selected_values = [multi_obj_values[i].tolist() for i in selected_indices]

    return next_pop, selected_indices, selected_values



def non_dominated_sorting(all_multi_obj_values):
    pop_size = len(all_multi_obj_values)  # 种群大小
    obj_count = len(all_multi_obj_values[0])  # 目标数量（如2个：延迟成本、仓储成本）

    # 初始化数据结构
    dominated_count = [0] * pop_size  # 每个个体被多少个其他个体支配
    dominate_list = [[] for _ in range(pop_size)]  # 每个个体支配的其他个体列表
    rank_list = [1] * pop_size  # 每个个体的层级（1级最优，数字越大越差）

    # 遍历每对个体，判断支配关系
    for i in range(pop_size):
        obj_i = all_multi_obj_values[i]  # 个体i的多目标值
        for j in range(pop_size):
            if i == j:
                continue  # 跳过自身
            obj_j = all_multi_obj_values[j]  # 个体j的多目标值

            # 判断：i是否支配j（最小化目标：i的所有目标≤j，且至少1个目标<j）
            is_dominate = True
            has_strict_better = False
            for k in range(obj_count):
                if obj_i[k] > obj_j[k]:  # 若i有一个目标比j差，直接不支配
                    is_dominate = False
                    break
                if obj_i[k] < obj_j[k]:  # 若i有一个目标比j好，标记为“严格更优”
                    has_strict_better = True

            if is_dominate and has_strict_better:
                dominate_list[i].append(j)  # i支配j，加入i的支配列表
                dominated_count[j] += 1  # j被支配次数+1

    # 按支配次数分层（1级→2级→...）
    current_rank = 1
    while True:
        # 找到当前层级的个体（被支配次数为0，且未分配层级）
        current_rank_individuals = [
            idx for idx in range(pop_size)
            if dominated_count[idx] == 0 and rank_list[idx] == 1
        ]
        if not current_rank_individuals:
            break  # 所有个体都分配完层级，退出

        # 为当前层级个体分配rank，并更新其支配个体的“被支配次数”
        for idx in current_rank_individuals:
            rank_list[idx] = current_rank
            # 该个体支配的所有个体，被支配次数-1（若减到0，下次会成为下一层级）
            for dominated_idx in dominate_list[idx]:
                dominated_count[dominated_idx] -= 1

        current_rank += 1  # 下一层级

    return rank_list, dominated_count


def calculate_crowding_distance(all_multi_obj_values, rank_list):
    pop_size = len(all_multi_obj_values)
    obj_count = len(all_multi_obj_values[0])
    crowding_distance_list = [0.0] * pop_size  # 初始化拥挤度为0

    # 按层级分组（同一层级的个体一起计算拥挤度）
    max_rank = max(rank_list)
    for rank in range(1, max_rank + 1):
        # 筛选当前层级的所有个体索引
        rank_individuals = [idx for idx in range(pop_size) if rank_list[idx] == rank]
        if len(rank_individuals) <= 2:
            # 若层级内个体数≤2，拥挤度设为无穷大（优先保留，避免边界解丢失）
            for idx in rank_individuals:
                crowding_distance_list[idx] = float('inf')
            continue

        # 对当前层级的个体，按每个目标维度排序
        for obj_idx in range(obj_count):
            # 按第obj_idx个目标值排序（升序，因为是最小化目标）
            sorted_by_obj = sorted(rank_individuals, key=lambda x: all_multi_obj_values[x][obj_idx])
            min_obj_val = all_multi_obj_values[sorted_by_obj[0]][obj_idx]  # 该目标的最小值
            max_obj_val = all_multi_obj_values[sorted_by_obj[-1]][obj_idx]  # 该目标的最大值

            # 边界个体（第一个和最后一个）拥挤度设为无穷大（避免边界解丢失）
            crowding_distance_list[sorted_by_obj[0]] = float('inf')
            crowding_distance_list[sorted_by_obj[-1]] = float('inf')

            # 中间个体：计算拥挤度（与相邻个体的目标差值占总范围的比例）
            for i in range(1, len(sorted_by_obj) - 1):
                prev_idx = sorted_by_obj[i - 1]
                curr_idx = sorted_by_obj[i]
                next_idx = sorted_by_obj[i + 1]
                # 拥挤度增量 = （后一个目标值 - 前一个目标值） / 目标值范围（避免量纲影响）
                if max_obj_val - min_obj_val == 0:
                    increment = 0.0  # 若目标值都相同，增量为0
                else:
                    increment = (all_multi_obj_values[next_idx][obj_idx] - all_multi_obj_values[prev_idx][obj_idx]) / (
                                max_obj_val - min_obj_val)
                crowding_distance_list[curr_idx] += increment

    return crowding_distance_list


# --------------------------
# 3. 第三步：种群筛选（生成下一代种群）
# 输入：population（原始种群）、all_multi_obj_values（所有个体的多目标值）、pop_size（目标种群大小）
# 输出：next_population（下一代种群）、next_multi_obj_values（下一代的多目标值）
# --------------------------
def select_next_population(population, all_multi_obj_values, pop_size):
    # 步骤1：非支配排序，得到每个个体的层级
    rank_list, _ = non_dominated_sorting(all_multi_obj_values)
    # 步骤2：计算每个个体的拥挤度
    crowding_distance_list = calculate_crowding_distance(all_multi_obj_values, rank_list)

    # 步骤3：按“层级优先（1级>2级>...），同层级拥挤度优先（大>小）”筛选个体
    # 生成个体索引列表，并按规则排序
    individual_indices = list(range(len(population)))
    # 排序规则：先按rank升序（层级优的在前），再按拥挤度降序（同层级多样性好的在前）
    individual_indices.sort(key=lambda x: (rank_list[x], -crowding_distance_list[x]))

    # 筛选前pop_size个个体作为下一代
    selected_indices = individual_indices[:pop_size]
    next_population = [population[idx] for idx in selected_indices]
    next_multi_obj_values = [all_multi_obj_values[idx] for idx in selected_indices]

    return next_population, next_multi_obj_values, rank_list, crowding_distance_list


def calculate_process_times(order, caigou_list, product_library, fluctuated_times, individual, purchase_process_counts, assembly_process_counts):
    # # print("---这是计算工序的开始时间与结束时间---")
    # # print("caigou_liat", caigou_list)
    # # print("purchase_process_counts", purchase_process_counts)
    # # print("assembly_process_counts", assembly_process_counts)
    # # print("product_library", product_library)
    # # print("fluctuated_times", fluctuated_times)
    # # print("individual", individual)
    purchase_end_times = [[] for _ in range(len(caigou_list))]
    # 初始化部装工序的开始和结束时间


    first_assembly_process = next(
        (process for product in product_library for process in product if process['工序类型'] == 1), None)

    if first_assembly_process is not None:
        # 计算该工序对应的加工机器数量
        num_assembly_machines = len(first_assembly_process['加工机器'])
    else:
        num_assembly_machines = 0
    # # print("num_assembly_machines",num_assembly_machines)

    sub_machine_operations = [0] * num_assembly_machines
    # # print("machine_operations",sub_machine_operations)



    # 计算采购工序结束时间
    for product_index, product_purchases in enumerate(caigou_list):
        product = product_library[product_index]
        purchase_processes = [p for p in product if p['工序类型'] == 0]
        for order_index, purchase in enumerate(product_purchases):
            end_time_list = []
            for process_index, supplier_index in enumerate(purchase):
                process = purchase_processes[process_index]
                # 获取原始采购时间并加上波动时间
                base_time = process['采购时间'][supplier_index]
                purchase_time = base_time + fluctuated_times[product_index][order_index][process_index]
                # 采购工序默认从0时刻开始
                start_time = 0
                end_time = start_time + purchase_time
                end_time_list.append(end_time)
            purchase_end_times[product_index].append(end_time_list)

    # # print("purchase_end_times", purchase_end_times)
    purchase_end_times_copy = copy.deepcopy(purchase_end_times)
    # print("purchase_end_times_copy", purchase_end_times_copy)
    # input("请任意键继续")


    # 计算部装工序时间
    internal_subassembly_sequence = individual["产品内部的部装工序顺序"]
    # print("产品内部的部装工序顺序", internal_subassembly_sequence)

    subassembly_machine_orders = individual["部装工序顺序列表"]
    # print("部装工序顺序列表", subassembly_machine_orders)

    subassembly_machine_choices = individual["部装机器选择列表"]
    # print("部装机器选择列表", subassembly_machine_choices)

    subassembly_list = [[[0,0]] * len(sublist) for sublist in internal_subassembly_sequence]
    # # print("各产品的部装的时间列表", subassembly_list)

    subassembly_end_time = []
    for order_num, operation_count in zip(order, assembly_process_counts):
        sub_list = []
        for _ in range(order_num):
            sub_sub_list = [0] * operation_count
            sub_list.append(sub_sub_list)
        subassembly_end_time.append(sub_list)

    subassembly_end_time_copy = copy.deepcopy(subassembly_end_time)
    # print("各类型产品的部装结束时间原始列表", subassembly_end_time)

    for i, subassembly_sequence in enumerate(subassembly_machine_orders):
        # # print("现在迭代的是列表中的第几个", i)
        # 这是这个工序对应的产品的编号 相同的数字表示了这个工序是哪个产品
        # # print("subassembly_sequence", subassembly_sequence)
        # 先判断这个工序对应产品是哪个类型的产品
        product_type = find_product_type(order, subassembly_sequence)
        # print("这是部装工序对应的产品类型", product_type)
        # 判断这个工序是产品中哪一道工序 具体是那一个工序要对应“产品内部的部装工序顺序”才能确定
        product_subassembly_process = get_occurrence_at_position(subassembly_machine_orders, i)
        # print("这是这个产品部装的第几个工序", product_subassembly_process)
        # 判断这个工序是产品中的哪一道工序
        product_process = int(internal_subassembly_sequence[subassembly_sequence][product_subassembly_process - 1])
        # print("这是这个产品的这一道工序", product_process)
        # 这是这个工序对应的前置工序
        product_process_info = product_library[product_type][product_process - 1]["前置工序"]
        # 这里的工序编号是从1开始的 所以要减去1
        product_process_info = [int(i) - 1 for i in product_process_info]
        # 这里的前置工序也是从1开始的 所以要减去1
        # print("这是这个工序对应的前置工序", product_process_info)
        # 例如这里是[2,3] 对应产品信息中的第3和第4个工序

        pro_purchase_process_info = []
        pro_subassembly_process_info = []

        for j, predecessor in enumerate(product_process_info):
            if product_library[product_type][predecessor]["工序类型"] == 0:
                pro_purchase_process_info.append(predecessor)
            if product_library[product_type][predecessor]["工序类型"] == 1:
                pro_subassembly_process_info.append(predecessor + 1)

        # print("这个工序的前置采购工序是", pro_purchase_process_info)
        # print("这个工序的前置部装工序是", pro_subassembly_process_info)

        # 找到这个最快能满足这个工序加工时间的部件的到站时间
        max_min1 = 0
        if len(pro_purchase_process_info) != 0:
            # print("purchase_end_times[产品类型]", purchase_end_times_copy[product_type])

            purchase_end_times_copy[product_type], max_min1 = find_and_modify_min_values(
                purchase_end_times_copy[product_type], pro_purchase_process_info)
            # print("修改后的purchase_end_times_copy[产品类型]", purchase_end_times_copy[product_type])
            # print("这个工序的开始时间1(采购部分)", max_min1)

            #这里是对的 考虑了同类产品工序的可替换性


        max_min2 = 0
        if len(pro_subassembly_process_info) != 0:
            # print("subassembly_end_time[产品类型]", subassembly_end_time[product_type])

            # print("这个工序的前置部装工序是", pro_subassembly_process_info)

            pro_subassembly_process_info = [i - purchase_process_counts[product_type] - 1 for i in pro_subassembly_process_info]

            # print("pro_subassembly_process_info", pro_subassembly_process_info)
            # 这是相对于把前置工序的编号转换为部装工序编码中列表的位置

            # print("输入是", subassembly_end_time_copy[product_type])

            subassembly_end_time_copy[product_type], max_min2 = find_and_modify_min_values2(
                subassembly_end_time_copy[product_type], pro_subassembly_process_info)

            # print("输出是", subassembly_end_time_copy[product_type])
            # print("输出是", max_min2)
            # print("修改后的subassembly_end_time[产品类型]", subassembly_end_time[product_type])
            # print("这个工序的开始时间2(部装部分)", max_min2)
            # 这就是找到一个列表集合里面 每一个列表是长度一样的列表 找到对应位置的最大的那个值


            # for k in pro_subassembly_process_info:
            #     # 首先判断这个部装工序是部装工序中的第几道
            #     a = k - purchase_process_counts[product_type] - 1
            #     b = subassembly_list[subassembly_sequence][a][1]
            #     if b > max_min2:
            #         max_min2 = b
            # # print("这个工序的开始时间2(部装部分)", max_min2)
            #这里 好像是没有考虑同种类型产品的工序替换性  所以需要修改

        max_min = max(max_min1, max_min2)

        machine_end_time = sub_machine_operations[subassembly_machine_choices[i]]
        # print("这是这个工序选择的机器结束时间", machine_end_time)

        max_min = max(max_min, machine_end_time)



        # print("这个工序的开始时间", max_min)

        # 这个工序的进行时间 依据这个工序选择的机器确定
        process_time = product_library[product_type][product_process - 1]["加工时间"][subassembly_machine_choices[i]]
        # print("这是这个工序的进行时间", process_time)
        # 这个工序的结束时间
        end_time = max_min + process_time
        # print("这是这个工序的结束时间", end_time)
        # 更新这个部装的时间列表
        # if end_time == float('inf'):
        #     input("这是这个工序的开始时间或结束时间为无穷大，请检查是否有前置工序未完成")

        # 更新使用机器的结束时间列表
        sub_machine_operations[subassembly_machine_choices[i]] = end_time


        rank = find_rank(internal_subassembly_sequence[subassembly_sequence], product_process)
        # print("这是这个工序是产品中第几个工序", rank)

        subassembly_list[subassembly_sequence][rank] = [max_min,end_time]
        # print("更新后的部装的时间列表", subassembly_list)
        # if max_min == float('inf') or end_time == float('inf'):
        #     input("这是这个工序的开始时间或结束时间为无穷大，请检查是否有前置工序未完成")

        # 更新这个部装结束时间列表
        aa = find_product_position(order, subassembly_sequence)
        subassembly_end_time[product_type][aa][rank] = end_time
        subassembly_end_time_copy[product_type][aa][rank] = end_time
        # 这相对于新完成的

    # print("更新后完整部装的时间列表",subassembly_list)
    subassembly_list_copy = copy.deepcopy(subassembly_list)

    # print("更新后完整部装时间列表", subassembly_end_time_copy)

    # print("更新后完整部装结束时间列表", subassembly_end_time)
    # print(len(subassembly_end_time))
    # 依次输出每个产品的部装结束时间
    # for product_type in range(len(subassembly_end_time)):
    #     print(f"产品类型 {product_type} 的部装结束时间列表:", subassembly_end_time[product_type])
    # input("请任意键继续")


    # 计算总装工序时间
    # print("subassembly_end_time_copy", subassembly_end_time_copy)
    #
    #
    #
    # input("暂停 看看部装完了 这个时间表里面有")



    final_assembly_process = next(
        (process for product in product_library for process in product if process['工序类型'] == 2), None)

    if final_assembly_process is not None:
        # 计算该工序对应的加工机器数量
        num_finalassembly_machines = len(final_assembly_process['加工机器'])
    else:
        num_finalassembly_machines = 0

    final_machine_operations = [0] * num_finalassembly_machines


    finalassembly_machine_orders = individual["总装工序顺序列表"]
    # print("总装工序顺序列表", finalassembly_machine_orders)

    finalassembly_machine_choices = individual["总装机器选择列表"]
    # print("总装机器选择列表", finalassembly_machine_choices)

    finalassembly_list = [[0, 0] for _ in finalassembly_machine_orders]
    # print("总装的时间列表", finalassembly_list)

    for j, finalassembly_sequence in enumerate(finalassembly_machine_orders):
        # print("现在迭代的是列表中的第几个", j)
        # 这是这个工序对应的产品的编号 相同的数字表示了这个工序是哪个产品
        # print("finalassembly_sequence", finalassembly_sequence)
        # 先判断这个工序对应产品是哪个类型的产品
        product_type = find_product_type(order, finalassembly_sequence)
        # print("这是总装工序对应的产品类型", product_type)
        # 总装一定是一个产品的最后一个工序

        # 这是这个工序对应的前置工序
        product_process_info = product_library[product_type][-1]["前置工序"]
        # 这里的工序编号是从1开始的 所以要减去1
        product_process_info = [int(i) - 1 for i in product_process_info]
        # 这里的前置工序也是从1开始的 所以要减去1
        # print("这是这个总装工序对应的前置工序", product_process_info)

        # 例如这里是[2,3] 对应产品信息中的第3和第4个工序

        pro_purchase_process_info = []
        pro_subassembly_process_info = []

        for x, predecessor in enumerate(product_process_info):
            if product_library[product_type][predecessor]["工序类型"] == 0:
                pro_purchase_process_info.append(predecessor)
            if product_library[product_type][predecessor]["工序类型"] == 1:
                pro_subassembly_process_info.append(predecessor + 1)

        # print("这个工序的前置采购工序是", pro_purchase_process_info)
        # print("这个工序的前置部装工序是", pro_subassembly_process_info)



        # 找到这个最快能满足这个工序加工时间的部件的到站时间
        max_min1 = 0
        if len(pro_purchase_process_info) != 0:
            # 如果这个总装工序的前置工序包含这个采购的原始零部件
            # print("purchase_end_times[产品类型]", purchase_end_times_copy[product_type])

            purchase_end_times_copy[product_type], max_min1 = find_and_modify_min_values(
                purchase_end_times_copy[product_type], pro_purchase_process_info)
            # print("修改后的purchase_end_times_copy[产品类型]", purchase_end_times_copy[product_type])
            # print("这个工序的开始时间1(采购部分)", max_min1)
            # if max_min1 == float('inf'):
            #     print("max_min1")
            #     input("采购工序有无限大")

        max_min2 = 0
        if len(pro_subassembly_process_info) != 0:
            # print("subassembly_end_time[产品类型]", subassembly_end_time[product_type])

            pro_subassembly_process_info = [i - purchase_process_counts[product_type] - 1 for i in
                                            pro_subassembly_process_info]

            # print("这个工序的前置部装工序是", pro_subassembly_process_info)
            #
            # print("输入是",subassembly_end_time_copy[product_type])

            subassembly_end_time_copy[product_type], max_min2 = find_and_modify_min_values2(
                subassembly_end_time_copy[product_type], pro_subassembly_process_info)
            # print("修改后的subassembly_end_time[产品类型]", subassembly_end_time[product_type])
            # print("这个工序的开始时间2(部装部分)", max_min2)

            # print("输出是",subassembly_end_time_copy[product_type])
            # print("输出是",max_min2)



            # for k in pro_subassembly_process_info:
            #     # 首先判断这个部装工序是部装工序中的第几道
            #     a = k - purchase_process_counts[product_type] - 1
            #     b = subassembly_list[finalassembly_sequence][a][1]
            #     if b > max_min2:
            #         max_min2 = b
            # print("这个工序的开始时间2(部装部分)", max_min2)
            # if max_min2 == float('inf'):
            #     print("max_min2")
            #     input("部装工序有无限大")

        max_min = max(max_min1, max_min2)

        machine_end_time = final_machine_operations[finalassembly_machine_choices[j]]

        # print("这是这个工序选择的机器", finalassembly_machine_choices[j])
        # print("这是这个总装工序选择的机器的结束时间", machine_end_time)

        max_min = max(max_min, machine_end_time)
        # print("这个工序的开始时间", max_min)
        # if max_min == float('inf'):
        #     print("max_min")
        #     input("这是这个工序的开始时间或结束时间为无穷大，请检查是否有前置工序未完成")

        # 这个工序的进行时间 依据这个工序选择的机器确定
        process_time = product_library[product_type][-1]["加工时间"][finalassembly_machine_choices[j]]
        # print("这是这个工序的进行时间", process_time)
        # 这个工序的结束时间
        end_time = max_min + process_time
        # print("这是这个工序的结束时间", end_time)

        # if end_time == float('inf'):
        #     print("end")
        #     input("这是这个工序的开始时间或结束时间为无穷大，请检查是否有前置工序未完成")

        # print("采购结束时应该全为inf", purchase_end_times_copy)

        # if max_min == float('inf') or end_time == float('inf'):
        #     input("这是这个工序的开始时间或结束时间为无穷大，请检查是否有前置工序未完成")
        # 更新这个部装的时间列表
        # 更新总装机器的结束时间列表
        final_machine_operations[finalassembly_machine_choices[j]] = end_time

        finalassembly_list[finalassembly_sequence] = [max_min, end_time]
        # print("更新后的部装的时间列表", finalassembly_list)

    # print("全部更新后的总装的时间列表", finalassembly_list)
    # 暂停
    # print("全部更新后的部装结束时间列表",subassembly_end_time)
    # print("全部更新后的部装结束时间列表",subassembly_list_copy)
    # input("暂停")


    return purchase_end_times, subassembly_list_copy, finalassembly_list





def draw_gantt_chart(purchase_end_times, prat_process_start_end_times, final_process_start_end_times, order):
    # 设置中文字体，解决乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制甘特图
    plt.figure(figsize=(12, 10))

    # 统一的工序间隔
    gap = 1

    # 计算每个产品的总工序数
    total_processes_per_product = [len(purchase) + 1 + 1 for purchase in purchase_end_times]

    # 计算所有产品的总工序数
    total_processes = sum(total_processes_per_product)

    # 计算每个产品的起始 y 位置
    product_start_y = [0]
    for i in range(len(total_processes_per_product) - 1):
        product_start_y.append(product_start_y[i] + total_processes_per_product[i])

    # 各类产品的起始索引
    product_type_start_idx = [0]
    for num in order[:-1]:
        product_type_start_idx.append(product_type_start_idx[-1] + num)

    # 用于存储每个产品的 y 轴中心位置
    product_y_centers = []

    # 定义不同工序类型的颜色
    purchase_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    part_process_color = '#8c564b'
    final_process_color = '#e377c2'

    for product_type in range(len(order)):
        for product_idx in range(order[product_type]):
            y_pos = product_start_y[product_type_start_idx[product_type] + product_idx]

            # 记录当前产品的所有工序的 y 轴位置
            all_y_positions = []

            # 计算当前产品的工序数量
            num_processes = len(purchase_end_times[product_type_start_idx[product_type] + product_idx]) + \
                            len(prat_process_start_end_times[product_type_start_idx[product_type] + product_idx]) + 1

            # 计算每个工序的高度
            process_height = 0.8

            # 绘制采购工序
            purchase_info = purchase_end_times[product_type_start_idx[product_type] + product_idx]
            for idx, end_time in enumerate(purchase_info):
                rect = plt.barh(y_pos, end_time, left=0, height=process_height, color=purchase_colors[idx % len(purchase_colors)])
                text_y = y_pos + process_height / 2
                plt.text(rect[0].get_width() / 2, text_y, f'采购工序{idx + 1}', ha='center', va='center', fontsize=6)
                all_y_positions.append(y_pos)
                y_pos += gap

            # 绘制工序类型 1（部转工序）
            part_process_info = prat_process_start_end_times[product_type_start_idx[product_type] + product_idx]
            for sub_idx, (start, end) in enumerate(part_process_info):
                rect = plt.barh(y_pos, end - start, left=start, height=process_height, color=part_process_color)
                text_y = y_pos + process_height / 2
                plt.text(start + (end - start) / 2, text_y, f'部装工序{sub_idx + 1}', ha='center', va='center', fontsize=6)
                all_y_positions.append(y_pos)
                y_pos += gap

            # 绘制工序类型 2（总装工序）
            start, end = final_process_start_end_times[product_type_start_idx[product_type] + product_idx]
            rect = plt.barh(y_pos, end - start, left=start, height=process_height, color=final_process_color)
            text_y = y_pos + process_height / 2
            plt.text(start + (end - start) / 2, text_y, f'总装工序', ha='center', va='center', fontsize=6)
            all_y_positions.append(y_pos)
            y_pos += gap

            # 计算当前产品所有工序的 y 轴中心位置
            product_y_center = sum(all_y_positions) / len(all_y_positions)
            product_y_centers.append(product_y_center)

    # 设置轴标签和标题
    plt.xlabel('时间')
    plt.ylabel('产品')
    plt.title('全部订单产品工序甘特图')

    # 设置 y 轴刻度
    y_ticks = []
    y_labels = []
    for product_type in range(len(order)):
        for product_idx in range(order[product_type]):
            y_ticks.append(product_y_centers[product_type_start_idx[product_type] + product_idx])
            y_labels.append(f"产品{product_type + 1}")

    plt.yticks(y_ticks, y_labels)

    # 显示图表
    plt.grid(True)
    plt.show()

# 交叉前：父代选择（多目标场景用「锦标赛选择」，兼顾成本与多样性）
def tournament_selection(population, all_multi_obj, tournament_size=3):
    # 随机选k个候选
    # # print("进入了锦标赛选择")
    candidates_idx = random.sample(range(len(population)), tournament_size)
    # # print("candidates_idx",candidates_idx)
    # # print(len(population))
    # # print(len(all_multi_obj))
    candidates = [(population[i], all_multi_obj[i], i) for i in candidates_idx]
    # # print("candidates",candidates)

    # # print(sum(x[1]))

    # 仅按综合成本升序排序（核心简化）
    candidates_sorted = sorted(candidates, key=lambda x: sum(x[1]))
    # # print("candidates_sorted",candidates_sorted)
    return candidates_sorted[0][0], candidates_sorted[0][1]  # 返回综合成本最低的候选作为父代


def cross_whole_part(parent1, parent2, part_def, crossover_rate):
    """
    处理单个部分的整体交叉（串行执行，无多进程）
    参数: 同原函数，去掉了多进程的args元组封装
    返回: (部分名称, {模块1: 子代1值, ...}, {模块1: 子代2值, ...})
    """
    part_name, modules = part_def
    # random.seed(random_seed)  # 保持独立随机种子，确保随机性

    # 决定该部分是否整体交叉
    if random.random() < crossover_rate:
        # 交叉：子代1用父2的模块，子代2用父1的模块
        child1_modules = {module: parent2[module] for module in modules}
        child2_modules = {module: parent1[module] for module in modules}
        # print(f"{part_name}发生整体交叉")
    else:
        # 不交叉：保留父代各自的模块
        child1_modules = {module: parent1[module] for module in modules}
        child2_modules = {module: parent2[module] for module in modules}

    return (part_name, child1_modules, child2_modules)


def serial_whole_crossover(parent1, parent2, crossover_rate):
    """
    四部分整体交叉（串行版，按顺序处理每个部分）
    功能与原parallel_whole_crossover完全一致，仅去掉多进程逻辑
    """
    # 复制父代生成子代基础（避免直接修改原父代数据）
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # 定义四部分结构（与原代码完全一致，保证交叉逻辑不变）
    parts_definition = [
        ("采购部分", ["采购供应商选择"]),
        ("部装组合部分", ["产品内部的部装工序顺序", "部装工序顺序列表", "部装机器选择列表"]),
        ("总装组合部分", ["总装工序顺序列表", "总装机器选择列表"])
    ]

    # 串行处理每个部分：按顺序遍历4个部分，逐个执行交叉
    for part_def in parts_definition:
        # 为每个部分生成独立随机种子（保持与原多进程版相同的随机性）
        # random_seed = random.getrandbits(32)
        # 调用交叉函数处理当前部分
        _, child1_modules, child2_modules = cross_whole_part(
            parent1, parent2, part_def, crossover_rate
        )
        # 合并当前部分的交叉结果到子代
        for module, value in child1_modules.items():
            child1[module] = value
        for module, value in child2_modules.items():
            child2[module] = value

    return child1, child2


def decode_individual(individual, orderss, product_library,purchase_process_counts, assembly_process_counts, order_delay_cost, product_Warehousing_Cost, order_Deadline):
    # 解码过程
    # # print("individual",individual)
    orders = merge_orders(orderss)
    # # print("orders",orders)
    # # print("product_library",product_library)
    # # print("purchase_process_counts",purchase_process_counts)
    # # print("assembly_process_counts",assembly_process_counts)



    caigou_list = individual["采购供应商选择"]
    # # print(caigou_list)
    caigou_ziliat = [[] for _ in range(len(orders))]

    purchase_cost = 0
    gene_index = 0
    for index, order in enumerate(orders):
        product_index = index
        quantity = order
        product = product_library[product_index]

        # 统计该产品的采购工序数量
        purchase_process_count = sum(1 for process in product if process['工序类型'] == 0)

        if quantity != 0:
            for _ in range(quantity):
                caigou_ziliat[product_index].append(caigou_list[:purchase_process_count])
                caigou_list = caigou_list[purchase_process_count:]

    # # print("caigou_ziliat", caigou_ziliat)


    # 在这里加上 部装子列表 和 总装子列表

    caigou_cost = computer_caigou_cost(caigou_ziliat, product_library)
    # # print("caigou_cost",caigou_cost)


    # 采购波动
    fluctuated_times = simulate_purchase_time_fluctuation(product_library, caigou_ziliat)
    # # print("fluctuated_times",fluctuated_times)


    # 计算工序开始和结束时间
    purchase_end_times, prat_process_start_end_times, final_process_start_end_times = calculate_process_times(orders, caigou_ziliat, product_library, fluctuated_times, individual, purchase_process_counts, assembly_process_counts)
    # purchase_end_times 是产品全部零部件采购的实际到站时间
    # prat_process_start_end_times是产品的部装工序的开始时间和结束时间
    # final_process_start_end_times是产品的部装工序的开始时间和结束时间
    # print("purchase_end_times",purchase_end_times)
    # print("len(purchase_end_times)",len(purchase_end_times))
    # 输出purchase_end_times的每个元素
    # for i, purchase_end_time in enumerate(purchase_end_times):
    #     print(f"purchase_end_times[{i}] = {purchase_end_time}")
    # print("prat_process_start_end_times",prat_process_start_end_times)
    # print("len(prat_process_start_end_times)",len(prat_process_start_end_times))
    # # 输出prat_process_start_end_times的每个元素
    # for i, prat_process_start_end_time in enumerate(prat_process_start_end_times):
    #     print(f"prat_process_start_end_times[{i}] = {prat_process_start_end_time}")
    # print("final_process_start_end_times",final_process_start_end_times)
    # print("len(final_process_start_end_times)",len(final_process_start_end_times))
    # # 输出final_process_start_end_times的每个元素
    # for i, final_process_start_end_time in enumerate(final_process_start_end_times):
    #     print(f"final_process_start_end_times[{i}] = {final_process_start_end_time}")

    # input("按任意键继续...")



    order_list, type_list = generate_product_mappings(orderss)

    # # print("产品编号对应的所属订单列表（索引=产品编号）：")
    # # print(order_list)
    # # print("\n产品编号对应的产品类型列表（索引=产品编号）：")
    # # print(type_list)


    # 计算总延迟成本和总仓储成本
    total_delay, total_warehousing = calculate_total_costs(final_process_start_end_times, order_list, type_list, order_delay_cost, order_Deadline, product_Warehousing_Cost)
    # # print("total_delay",total_delay)
    # # print("total_warehousing",total_warehousing)



    # input("暂停")
    #
    # new_purchase_end_times = []
    # for sublist in purchase_end_times:
    #     for inner_sublist in sublist:
    #         new_purchase_end_times.append(inner_sublist)


    # # 暂停
    # input("按任意键继续...")
    #
    # for i, sublist in enumerate(purchase_end_times):
    #     # print("第", i, "类产品的零部件实际到站时间", sublist)
    # # print("--------------")
    # for i, sublist in enumerate(prat_process_start_end_times):
    #     # print("第", i, "类产品的部装工序的开始时间和结束时间", sublist)
    # # print("--------------")
    # for i, sublist in enumerate(final_process_start_end_times):
    #     # print("第", i, "类产品的部装工序的开始时间和结束时间", sublist)




    # draw_gantt_chart(new_purchase_end_times, prat_process_start_end_times, final_process_start_end_times, orders)
    # 找到总装工序结束时间最大的那个值
    # max_second_value = float('-inf')
    # for sub_list in final_process_start_end_times:
    #     if sub_list[1] > max_second_value:
    #         max_second_value = sub_list[1]
    # # print("max_final_process_end_time",max_second_value)

    # # 暂停
    # input("按任意键继续...")
    return [caigou_cost, total_delay, total_warehousing]


def analyze_processes(processes):
    process_info = []
    for process in processes:
        operation_name = process[1]
        predecessors = process[6].split(',') if process[6] else []
        if '采购' in operation_name:
            process_type = 0
            suppliers = process[2].split(',') if process[2] else []
            costs = [float(x) for x in process[3].split(',')] if process[3] else []
            purchase_times = [float(x) for x in process[4].split(',')] if process[4] else []
            process_info.append({
                '工序编号': process[0],
                '工序类型': process_type,
                '供应商': suppliers,
                '单位成本': costs,
                '采购时间': purchase_times,
                '前置工序': predecessors
            })
        elif '部装' in operation_name:
            process_type = 1
            machines = process[2].split(',') if process[2] else []
            processing_times = [float(x) for x in process[5].split(',')] if process[5] else []
            process_info.append({
                '工序编号': process[0],
                '工序类型': process_type,
                '加工机器': machines,
                '加工时间': processing_times,
                '前置工序': predecessors
            })
        elif '总装' in operation_name:
            process_type = 2
            machines = process[2].split(',') if process[2] else []
            processing_times = [float(x) for x in process[5].split(',')] if process[5] else []
            process_info.append({
                '工序编号': process[0],
                '工序类型': process_type,
                '加工机器': machines,
                '加工时间': processing_times,
                '前置工序': predecessors
            })
    return process_info


def plan_order(orders, product_library):
    # orders是订单
    # product_library是产品库，包含每一个产品的采购、部装和总装的信息
    if not product_library:
        # print("错误：产品库为空。")
        return 0, 0, []

    total_cost = 0
    total_time = 0
    plan = []

    for order in orders:
        product_index = order[0] - 1  # 产品索引从 0 开始
        # 产品的编号
        quantity = order[1]
        # 编号product_index产品的数量

        # 检查产品索引是否有效
        if product_index < 0 or product_index >= len(product_library):
            # print(f"错误：订单中的产品编号 {order[0]} 超出范围。")
            continue

        product = product_library[product_index]
        # # print(product)
        # # print("product")
        # print("产品编号",product_index)
        # 处理采购工序
        for process in product:
            # print("工序编号",process["工序编号"])

            # 这是对于产品的每一个工序进行处理
            if process['工序类型'] == 0:
                min_cost = float('inf')
                min_time = float('inf')
                best_supplier_index = 0

                for i in range(len(process['供应商'])):
                    cost = process['单位成本'][i] * quantity
                    time = process['采购时间'][i]
                    if cost < min_cost or (cost == min_cost and time < min_time):
                        min_cost = cost
                        min_time = time
                        best_supplier_index = i

                total_cost += min_cost
                total_time = max(total_time, min_time)
                plan.append({
                    '产品编号': product_index + 1,
                    '工序编号': process['工序编号'],
                    '操作类型': '采购',
                    '供应商': process['供应商'][best_supplier_index],
                    '成本': min_cost,
                    '时间': min_time
                })

        # 处理加工工序
        for process in product:
            if process['工序类型'] in [1, 2]:
                min_time = float('inf')
                best_machine_index = 0

                for i in range(len(process['加工机器'])):
                    time = process['加工时间'][i] * quantity
                    if time < min_time:
                        min_time = time
                        best_machine_index = i

                total_time += min_time
                plan.append({
                    '产品编号': product_index + 1,
                    '工序编号': process['工序编号'],
                    '操作类型': '加工',
                    '机器': process['加工机器'][best_machine_index],
                    '时间': min_time
                })

    return total_cost, total_time, plan

def get_half_elite_individuals(elite_individuals, data):
    """
    从多目标值列表中筛选总成本最小的一半个体
    输入:
        data: List[List[float]]，每个子列表包含三个目标值
    输出:
        half_elite_individuals: 总成本最小的一半个体的三目标值列表
        half_vals_elite: 对应的总成本列表（即三目标值之和）
    """
    if not isinstance(data, list) or not all(isinstance(row, list) and len(row) == 3 for row in data):
        raise ValueError("输入必须是一个包含三个浮点值的二维列表")

    # 计算总成本
    total_costs = [sum(row) for row in data]

    # 找出总成本最小的一半的索引
    sorted_indices = np.argsort(total_costs)
    half = len(data) // 2
    elite_indices = sorted_indices[:half].tolist()

    # 提取对应个体和其多目标值
    half_elite_individuals = [elite_individuals[i] for i in elite_indices]
    half_vals_elite = [data[i] for i in elite_indices]

    return half_elite_individuals, half_vals_elite

def genetic_algorithm(orderss, product_library, purchase_process_counts, assembly_process_counts, order_delay_cost, product_Warehousing_Cost, order_Deadline, population_size=50, generations=100, crossover_rate=0.8, mutation=0.1, llm_guided_ratio=LLM_GUIDED_RATIO):

    order_list, type_list = generate_product_mappings(orderss)

    # print("order_list",order_list)
    # print("type_list",type_list)
    purchase_ranges = get_purchase_ranges(type_list, product_library)
    # print("purchase_ranges",purchase_ranges)
    # input("暂停")
    type_counts = build_assembly_count_list(product_library, type_list)


    orders = merge_orders(orderss)

    population = generate_population(population_size, orders, product_library)
    # print("population0",population[0])
    # print("population1",population[1])
    # input("暂停")

    # 存储所有个体的多目标值（索引对应个体在种群中的位置）
    all_multi_objective_values = []

    for i, individual in enumerate(population):
        evaluations = []

        for _ in range(30):  # 评估 30 次
            multi_objective_values = decode_individual(
                individual,
                orderss,
                product_library,
                purchase_process_counts,
                assembly_process_counts,
                order_delay_cost,
                product_Warehousing_Cost,
                order_Deadline
            )
            evaluations.append(multi_objective_values)

        # 将 30 次评估结果转为 numpy 数组，按列取均值
        evaluations_array = np.array(evaluations)  # shape: (30, 3)
        averaged_values = np.mean(evaluations_array, axis=0)  # shape: (3,)

        # 添加到最终列表
        all_multi_objective_values.append(averaged_values.tolist())


    aaa, ___, CCC = dynamic_select_next_generation(population, all_multi_objective_values, ELITE_RATIO)

    # 记录每一代的一些优解（用于大模型分析给出新的子代）
    history_pareto = []
    history_multi_obj = []
    evaluation_thi_final = []
    history_pareto.append([aaa, CCC])
    history_multi_obj.append(all_multi_objective_values)


    cgl = []

    for gen in range(generations):
        # print(f"=== 第{gen + 1}/{generations}代迭代 ===")

        # if gen == 1:
        #     first_gen_records = logger.get_last_generation_records(0)
        #     # for rec in first_gen_records:
        #     #     print(rec)
        #
        #     # print("个体",population[0])
        #
        #     top10 = find_top_k_similar_max_or_avg(population[0], first_gen_records, k=10)
        #     # print("top20[0]",top20[0])
        #
        #     # for item in top20:
        #     #     print(f"最终相似度: {item['similarity']:.4f} | 最大: {item['max_similarity']:.4f} | 平均: {item['avg_similarity']:.4f}")
        #     #     print("适应度:", item["record"].get("fitness_mutated", None))
        #     #     print("-" * 50)
        #
        #     # records = extract_structured_evolution_records(top20)
        #     # print("20个交叉变异记录",records)
        #     formatted_text = format_crossover_mutation_records(top10)
        #     # print("20个交叉变异记录嵌入提示词的清晰结构", formatted_text)
        #
        #     supplier_context = build_prompt_for_supplier_llm(population[0], formatted_text)
        #     print("采购供应商选择段编码",population[0]['采购供应商选择'])
        #
        #
        #
        #     supplieragent = SupplierAgent()
        #     result = supplieragent.generate(supplier_context)
        #     print("完整采购编码段提示词",result)
        #
        #     # print("采购供应商选择段编码长度",len(population[0]['采购供应商选择']))
        #
        #
        #
        #     input("第1代暂停")
        #
        #
        #     print("✅ 生成的采购供应商选择段编码：")
        #     print(result)




        # 1. 筛选当前代的10%精英个体
        elite_individuals, idx_elite, vals_elite = dynamic_select_next_generation(population, all_multi_objective_values, ELITE_RATIO)
        # print("精英个体",elite_individuals)
        # print("精英个体索引",idx_elite)
        # print("精英个体多目标值",vals_elite)
        # print(population[idx_elite[0]])
        # input("第1代精英个体暂停")

        # 找到elite_individuals一半的总成本最小的个体和对应的多目标值
        half_elite_individuals, half_vals_elite = get_half_elite_individuals(elite_individuals, vals_elite)
        # print("精英个体一半的成本最小个体",half_elite_individuals)
        # print("精英个体一半的成本最小多目标值",half_vals_elite)
        # input("第1代精英个体一半暂停")

        elite_size = len(elite_individuals)  # 理论上=population_size*0.1（如100→10）

        print(f"第{gen + 1}代精英个体多目标值: {vals_elite}")

        # 输出最小的成本
        min_total = min(sum(x) for x in vals_elite)
        print(f"最小成本: {min_total}")

        # print("精英个体",elite_individuals)
        # print(len(elite_individuals))
        # print("vals_elite",vals_elite)
        re_num_llm = 0
        llm_result = []
        if gen > round(generations * 0):
            num_llm = int(round(len(population) * llm_guided_ratio))
            if num_llm == 0:
                re_num_llm = 0

            if num_llm != 0:
                context = build_prompt_for_llm(half_elite_individuals, half_vals_elite, num_llm, type_list, purchase_ranges, type_counts)
                # print(context)
                # input("第1代大模型提示词暂停")

                individual_agent = individualAgent()

                llm_result = individual_agent.generate(
                    context,
                    product_library,
                    orderss,
                    order_delay_cost,
                    product_Warehousing_Cost,
                    order_Deadline,
                    type_list,
                    purchase_ranges,
                    type_counts,
                    generation=gen,
                    logical_call_index=1,
                    requested_candidates=num_llm,
                    elite_context_size=len(half_elite_individuals),
                )
                print("大模型生成的个体",llm_result)



                re_num_llm = len(llm_result)
                print(f"大模型生成的个体成功率: {re_num_llm / num_llm:.4f}")
                cgl.append(re_num_llm / num_llm)
        else:
            re_num_llm = 0

        # if gen == 10:
        #     # 输出全部十次平均成功率
        #     print(f"10代平均成功率: {sum(cgl) / len(cgl):.4f}")
        #     input("第10代暂停")




        # print("大模型生成的个体",result)
        # 检查大模型生成的采购供应商选择段编码是否合法
        # print("大模型生成的个体数",len(result))
        #
        # input("第1代暂停")

        # 2. 交叉生成新个体（补充剩余非精英个体和非大模型生成个体，保持种群大小）
        # 2.1 计算需要生成的新个体数量（90%）
        need_new_count = population_size - elite_size - re_num_llm
        # 2.2 步骤2：
        # --------------------------
        # 需生成的新个体数量 = 总种群大小 - 精英数量（如100-10=90）
        new_individuals = []


        while len(new_individuals) < need_new_count:
            # 3.1 选2个父代（用锦标赛选择）

            parent1, f_p1 = tournament_selection(population, all_multi_objective_values)
            parent2, f_p2 = tournament_selection(population, all_multi_objective_values)

            # # print("parent1",parent1)
            # # print("parent2",parent2)
            # start = time.time()

            child1, child2 = serial_whole_crossover(parent1, parent2, crossover_rate)

            child1_objective_values = decode_individual(
                child1,
                orderss,
                product_library,
                purchase_process_counts,
                assembly_process_counts,
                order_delay_cost,
                product_Warehousing_Cost,
                order_Deadline
            )
            child2_objective_values = decode_individual(
                child2,
                orderss,
                product_library,
                purchase_process_counts,
                assembly_process_counts,
                order_delay_cost,
                product_Warehousing_Cost,
                order_Deadline
            )
            # end = time.time()
            # # print("选择交叉时间", end - start)
            # # print("child1_objective_values",child1_objective_values)
            # # print("child2_objective_values",child2_objective_values)
            # # print(sum(child1_objective_values))
            # # print(sum(child2_objective_values))
            if sum(child1_objective_values) < sum(child2_objective_values):
                new_individuals.append(child1)
            else:
                new_individuals.append(child2)

            logger.log_crossover(
                generation=gen,
                p1=parent1, p2=parent2,
                c1=child1, c2=child2,
                f_p1=f_p1, f_p2=f_p2,
                f_c1=child1_objective_values, f_c2=child2_objective_values,
            )

        # 对所有新个体执行变异
        mutated_new_individuals = []
        for child in new_individuals:
            childcopy = copy.deepcopy(child)

            childcopy_objective_values = decode_individual(
                childcopy,
                orderss,
                product_library,
                purchase_process_counts,
                assembly_process_counts,
                order_delay_cost,
                product_Warehousing_Cost,
                order_Deadline
            )

            # 采购段执行变异
            if random.random() < mutation:
                purchase_chromosome = child['采购供应商选择']
                # # print(purchase_chromosome)

                # 执行变异（仅返回变异后的染色体）
                purchasemutated_chrom = purchase_mutate_chromosome(
                    chromosome=purchase_chromosome,
                    product_types=type_list,
                    proc_counts=purchase_process_counts,
                    product_library=product_library,
                    mutation_rate=0.3
                )
                child['采购供应商选择'] = purchasemutated_chrom
                # # print(child['采购供应商选择'])

            # 产品内部的部装工序顺序变异
            assembly_chromosome = child['产品内部的部装工序顺序']

            assemblymutated_chrom, changes = mutate_assembly_orders(product_library, type_list, assembly_chromosome, mutation_rate=0.2)

            child['产品内部的部装工序顺序'] = assemblymutated_chrom

            # 产品内部的部装工序顺序变异会引起这个 部装工序加工机械选择列表的变异 需要进行对应的调整才能是合法的解
            # print("change位置",changes)

            assembly_process_sequence_chromosome = child['部装工序顺序列表']
            process_machine_selection_chromosome = child['部装机器选择列表']

            child['部装机器选择列表'] = swap_machine_selection(changes,assembly_process_sequence_chromosome,process_machine_selection_chromosome)

            # print("assembly_process_sequence_chromosome",assembly_process_sequence_chromosome)
            # print("process_machine_selection_chromosome",process_machine_selection_chromosome)

            #产品全局部装顺序和对应机器的变异
            if random.random() < mutation:
            # if random.random() < 1:
                new_assembly, new_machine = mutate_with_machine_remap_positions(
                    assembly_process_sequence_chromosome,
                    process_machine_selection_chromosome,
                    min_ratio=0.1,
                    max_ratio=0.3
                )
                child['部装工序顺序列表'] = new_assembly
                child['部装机器选择列表'] = new_machine

            # 机器对应列表的变异

            # # print("产品内部的部装工序顺序",child['产品内部的部装工序顺序'])
            # # print("部装工序顺序列表",child['部装工序顺序列表'])
            # # print("部装机器选择列表",child['部装机器选择列表'])
            # # print("type_list",type_list)


            # 部装工序机器选择的变异
            if random.random() < mutation:
                new_machine_seq = mutate_machine_selection_multi_dynamic(
                    child['部装机器选择列表'],
                    child['部装工序顺序列表'],
                    child['产品内部的部装工序顺序'],
                    type_list,
                    product_library,
                    min_ratio=0.05,  # 最少 5%
                    max_ratio=0.2  # 最多 20%
                )
                child['部装机器选择列表'] = new_machine_seq
                # # print(child['部装机器选择列表'])

            # # print(child["总装工序顺序列表"])
            # # print(child["总装机器选择列表"])

            # 总装工序顺序列表的变异 和 对应总装工序机器选择的调整
            if random.random() < mutation:
                new_proc, new_mach = mutate_shuffle_positions_dual(
                    child["总装工序顺序列表"],
                    child["总装机器选择列表"],
                    min_ratio=0.1,  # 最少 20% 位置
                    max_ratio=0.3  # 最多 50% 位置
                )
                child["总装工序顺序列表"] = new_proc
                child["总装机器选择列表"] = new_mach


            # 总装工序机器选择的变异

            if random.random() < mutation:
                new_machine_seq = mutate_final_assembly_machine_selection(
                    child["总装工序顺序列表"],
                    child["总装机器选择列表"],
                    type_list,
                    product_library,
                    min_ratio=0.2,
                    max_ratio=0.4
                )
                child["总装机器选择列表"] = new_machine_seq

            mutated_new_individuals.append(child)

            child_objective_values = decode_individual(
                child,
                orderss,
                product_library,
                purchase_process_counts,
                assembly_process_counts,
                order_delay_cost,
                product_Warehousing_Cost,
                order_Deadline
            )

            # logger.log_mutation(childcopy, childcopy_objective_values, child, child_objective_values)

        next_population = elite_individuals + llm_result + mutated_new_individuals
        next_population = next_population[:population_size]  # 冗余截断（理论上不会触发）
        print("这一代种群的大小",len(next_population))

        next_all_multi_obj = []

        for ind in next_population:
            evaluations = []

            for _ in range(30):  # 评估 30 次
                multi_objective_values = decode_individual(
                    ind,
                    orderss,
                    product_library,
                    purchase_process_counts,
                    assembly_process_counts,
                    order_delay_cost,
                    product_Warehousing_Cost,
                    order_Deadline
                )
                evaluations.append(multi_objective_values)

            # 转为 numpy 数组并取均值
            if gen == generations - 1:
                evaluation_thi_final.append(evaluations)
            evaluations_array = np.array(evaluations)  # shape: (30, 3)
            averaged_values = np.mean(evaluations_array, axis=0)  # shape: (3,)
            next_all_multi_obj.append(averaged_values.tolist())

        NEXT_elite_individuals, idx_elite, vals_elite = dynamic_select_next_generation(next_population, next_all_multi_obj, ELITE_RATIO)

        # print("NEXT_elite_individuals",NEXT_elite_individuals)
        # print("vals_elite",vals_elite)
        # input("暂停")

        history_pareto.append([NEXT_elite_individuals,vals_elite])
        history_multi_obj.append(next_all_multi_obj)

        population = next_population
        all_multi_objective_values = next_all_multi_obj

        # current_min_cost = min(sum(obj) for obj in next_all_multi_obj)
        # print(f"当前代综合成本最小值：{current_min_cost:.2f}")

    # print("\n=== 遗传算法迭代结束 ===")
    final_pareto = sorted(
        zip(population, all_multi_objective_values),
        key=lambda x: sum(x[1])
    )[:10]

    # 输出最终最优解的信息
    # print("最终10个最优个体的多目标值（成本1, 成本2, 成本3）：")
    for i, (ind, obj) in enumerate(final_pareto, 1):
        print(f"最优个体{i}：综合成本={sum(obj):.2f}，多目标值={obj}")

    # 返回关键结果（可按需调整）
    return {
        "final_population": population,  # 最终代种群
        "final_multi_obj": all_multi_objective_values,  # 最终代多目标值
        "final_pareto": final_pareto,  # 最终帕累托最优解
        "history_pareto": history_pareto,  # 每代最优解历史（用于画图分析）
        "history_multi_obj": history_multi_obj,  # 每代多目标值历史（用于画图分析）
        "EVa_thi_final": evaluation_thi_final  # 最终代多目标值30次评估
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the AEEA DeepSeek experiment for the super-large WN4 instance."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_ALGORITHM_SEED)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--llm-guided-ratio", type=float, default=LLM_GUIDED_RATIO)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Unique run label. The default is a UTC timestamp.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    return parser.parse_args()


def main():
    global logger
    args = parse_arguments()
    if not 0 <= args.llm_guided_ratio <= 1:
        raise ValueError("--llm-guided-ratio must be between 0 and 1")
    if args.population_size < 3 or args.generations < 1:
        raise ValueError("population size must be at least 3 and generations at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    logger = EvolutionLogger()

    product_library = Product_library
    purchase_process_counts = [
        sum(1 for process in product if process['工序类型'] == 0)
        for product in product_library
    ]
    assembly_process_counts = [
        sum(1 for process in product if process['工序类型'] == 1)
        for product in product_library
    ]
    orderss = ordersss
    order_delay_cost = order_delay_costss
    product_Warehousing_Cost = product_Warehousing_Costss
    order_Deadline = order_Deadliness

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = (args.output_root / run_id).resolve()
    project_root = Path(__file__).resolve().parent
    source_files = [
        project_root / "run_experiment.py",
        project_root / "config3.py",
        project_root / "purchase_mutation.py",
        project_root / "plt_history.py",
        project_root / "evolution_logger.py",
        project_root / "product_information" / "setup_matrix.py",
        project_root / "mutil_agent_generation_individual" / "DeepSeek.py",
        project_root / "mutil_agent_generation_individual" / "llm_agent.py",
        project_root / "mutil_agent_generation_individual" / "prompt_builder3.py",
        project_root / "mutil_agent_generation_individual" / "output_check2.py",
        project_root / "mutil_agent_generation_individual" / "theoretical_lower_bound.py",
        project_root / "prompts" / "system_prompt.txt",
        project_root / "schemas" / "individuals.schema.json",
        project_root / "pricing.json",
        *sorted((project_root / "paper_six_wn4").glob("*.csv")),
    ]
    configure_reproducibility(
        run_dir=run_dir,
        run_metadata={
            "instance": "Set_D-WN4 (super-large)",
            "orders": orderss,
            "order_delay_cost": order_delay_cost,
            "product_warehousing_cost": product_Warehousing_Cost,
            "order_deadlines": order_Deadline,
            "population_size": args.population_size,
            "generations": args.generations,
            "crossover_rate": args.crossover_rate,
            "mutation_rate": args.mutation_rate,
            "elite_retention_ratio": ELITE_RATIO,
            "llm_guided_ratio": args.llm_guided_ratio,
            "evolutionary_offspring_ratio": 1 - ELITE_RATIO - args.llm_guided_ratio,
            "algorithm_random_seed": args.seed,
            "numpy_random_seed": args.seed,
            "llm_calls_per_generation": "1 for generations 1..T-1; 0 for generation 0",
            "candidates_requested_per_call": int(round(args.population_size * args.llm_guided_ratio)),
            "maximum_api_attempts_per_logical_call": 5,
            "objective_evaluations_per_individual": 30,
            "parsing": "json.loads followed by extraction of the top-level individuals array",
            "feasibility_handling": "six-segment validation and repair in output_check2.check_response",
        },
        llm_config=llm_configuration(),
        pricing_file=project_root / "pricing.json",
        schema_file=project_root / "schemas" / "individuals.schema.json",
        source_files=source_files,
    )

    try:
        result = genetic_algorithm(
            orderss,
            product_library,
            purchase_process_counts,
            assembly_process_counts,
            order_delay_cost,
            product_Warehousing_Cost,
            order_Deadline,
            population_size=args.population_size,
            generations=args.generations,
            crossover_rate=args.crossover_rate,
            mutation=args.mutation_rate,
            llm_guided_ratio=args.llm_guided_ratio,
        )
        with (run_dir / "result.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "EVa_thi_final": result["EVa_thi_final"],
                    "history_multi_obj": result["history_multi_obj"],
                    "final_population": result["final_population"],
                    "final_multi_obj": result["final_multi_obj"],
                },
                handle,
            )
        plot_pareto_trend_subplots(
            result["history_pareto"],
            save_path=str(run_dir / "pareto_trend.png"),
        )
        print(f"Run artifacts saved to: {run_dir}")
    finally:
        finalize_reproducibility()


if __name__ == "__main__":
    main()

#     individual = {
#         "采购供应商选择": individual1,
#         "产品内部的部装工序顺序": product_internal_sequences,
#         "部装工序顺序列表": process_sequence_list,
#         "部装机器选择列表": machine_choice_list,
#         "总装工序顺序列表": final_process_sequence_list,
#         "总装机器选择列表": final_machine_choice_list
#     }

