"""
风电领域指令微调数据集生成器
生成三类数据：
1. 解释类（最重要）
2. 因果推理类
3. 异常分析类
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
from typing import List, Dict
import random

try:
    from config import PROJECT_ROOT
    _DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "features" / "data_feature_layer.csv")
except ImportError:
    _DEFAULT_DATA_PATH = "data_feature_layer.csv"


class WindPowerInstructionGenerator:
    """风电指令数据集生成器"""
    
    def __init__(self, data_path: str = _DEFAULT_DATA_PATH):
        """
        参数:
            data_path: 特征数据路径
        """
        self.data = pd.read_csv(data_path)
        self.instructions = []
        
        print(f"✅ 加载数据: {self.data.shape}")
    
    def generate_type1_explanation(self, n_samples: int = 100) -> List[Dict]:
        """
        类型1：解释类指令数据
        
        参数:
            n_samples: 生成样本数
            
        返回:
            指令数据列表
        """
        print(f"\n📝 生成类型1：解释类指令数据 ({n_samples}条)...")
        
        instructions = []
        
        # 从数据中随机采样
        samples = self.data.sample(n=min(n_samples, len(self.data)))
        
        for idx, row in samples.iterrows():
            wind_speed = row['wind_speed']
            temperature = row['temperature']
            pressure = row['pressure']
            density = row['density']
            wind_power = row['wind_power']
            
            # 获取季节
            season = ''
            for col in self.data.columns:
                if col.startswith('season_') and row[col] == 1:
                    season = col.replace('season_', '')
                    break
            
            # 构建输入
            input_text = f"当前气象条件：风速{wind_speed:.1f}m/s，温度{temperature-273.15:.1f}℃，气压{pressure/100:.0f}hPa，空气密度{density:.3f}kg/m³，季节{season}。"
            
            # 构建输出（专家级解释）
            output_text = self._generate_expert_explanation(
                wind_speed, temperature, pressure, density, wind_power, season
            )
            
            instructions.append({
                'instruction': '请分析当前气象条件对风电功率的影响。',
                'input': input_text,
                'output': output_text,
                'type': 'explanation'
            })
        
        print(f"✅ 生成{len(instructions)}条解释类数据")
        return instructions
    
    def _generate_expert_explanation(
        self, 
        wind_speed: float,
        temperature: float,
        pressure: float,
        density: float,
        wind_power: float,
        season: str
    ) -> str:
        """生成专家级解释"""
        
        explanations = []
        
        # 1. 风速分析
        if wind_speed < 3:
            explanations.append(
                f"风速{wind_speed:.1f}m/s低于风机切入风速（3m/s），风机无法启动，预计功率接近零。"
            )
        elif 3 <= wind_speed < 8:
            explanations.append(
                f"风速{wind_speed:.1f}m/s处于低速区间，风机可运行但效率较低。根据风能公式P∝v³，功率约为{wind_power:.2f}kW。"
            )
        elif 8 <= wind_speed <= 15:
            explanations.append(
                f"风速{wind_speed:.1f}m/s处于最佳发电区间[8-15m/s]，风机运行效率高。当前预测功率{wind_power:.2f}kW，利用率接近最优。"
            )
        elif 15 < wind_speed < 25:
            explanations.append(
                f"风速{wind_speed:.1f}m/s超过额定风速，风机功率趋于饱和。虽然风速持续增加，但功率增长有限，当前约{wind_power:.2f}kW。"
            )
        else:
            explanations.append(
                f"风速{wind_speed:.1f}m/s超过切出风速（25m/s），风机将自动停机以保护设备，功率为零。"
            )
        
        # 2. 密度与温度分析
        temp_celsius = temperature - 273.15
        if density > 1.28:
            explanations.append(
                f"空气密度{density:.3f}kg/m³高于标准值（1.225kg/m³），主要由于低温（{temp_celsius:.1f}℃）和高气压（{pressure/100:.0f}hPa）共同作用。高密度空气可提升风能捕获效率约{(density-1.225)/1.225*100:.1f}%。"
            )
        elif density < 1.17:
            explanations.append(
                f"空气密度{density:.3f}kg/m³低于标准值，可能受高温（{temp_celsius:.1f}℃）或低气压影响。低密度降低单位体积空气动能，发电效率下降约{(1.225-density)/1.225*100:.1f}%。"
            )
        else:
            explanations.append(
                f"空气密度{density:.3f}kg/m³接近标准值，温度{temp_celsius:.1f}℃和气压{pressure/100:.0f}hPa处于常规范围，对发电效率影响中性。"
            )
        
        # 3. 季节性分析
        season_analysis = {
            '春': '春季气流不稳定，风速变化较大，需注意功率波动。',
            '夏': '夏季高温导致空气密度降低，整体发电效率偏低。',
            '秋': '秋季风况较为稳定，是风电开发的较好季节。',
            '冬': '冬季冷高压系统带来稳定强风，空气密度大，是最佳发电季节。'
        }
        
        if season in season_analysis:
            explanations.append(season_analysis[season])
        
        # 4. 综合评估
        theoretical_power = 0.5 * density * (wind_speed ** 3) * 0.4
        if abs(wind_power - theoretical_power) / theoretical_power < 0.2:
            explanations.append(
                f"综合评估：当前预测功率{wind_power:.2f}kW与理论值{theoretical_power:.2f}kW相符，预测置信度高。"
            )
        else:
            explanations.append(
                f"综合评估：当前预测功率{wind_power:.2f}kW与理论值{theoretical_power:.2f}kW存在偏差，建议结合实际监测数据验证。"
            )
        
        return ' '.join(explanations)
    
    def generate_type2_causality(self, n_samples: int = 50) -> List[Dict]:
        """
        类型2：因果推理类指令数据
        
        参数:
            n_samples: 生成样本数
            
        返回:
            指令数据列表
        """
        print(f"\n📝 生成类型2：因果推理类指令数据 ({n_samples}条)...")
        
        # 预定义因果问答对
        causality_pairs = [
            {
                'question': '为什么风速增加会提高风电功率？',
                'answer': '因为风能与风速的三次方成正比。根据风能公式P = 0.5 × ρ × A × v³ × Cp，当风速v增加时，功率P以立方关系增长。例如，风速从5m/s增加到10m/s，功率理论上增加8倍（2³=8）。这是风电系统最核心的物理规律。'
            },
            {
                'question': '温度变化如何影响风电功率？',
                'answer': '温度主要通过影响空气密度间接作用于风电功率。根据理想气体状态方程ρ = P/(R×T)，温度T降低时，空气密度ρ增大。冬季低温使空气密度比夏季高约10-15%，从而提升相同风速下的发电功率。这也是冬季风电场发电量通常高于夏季的主要原因。'
            },
            {
                'question': '为什么冬季风电效率比夏季高？',
                'answer': '冬季风电效率高主要有三个原因：1）冷高压系统带来更强更稳定的风力，平均风速高于夏季；2）低温使空气密度增加10-15%，提升风能捕获效率；3）冬季风况稳定性好，功率波动小。综合这三点，冬季风电场的容量因子通常比夏季高20-30%。'
            },
            {
                'question': '气压变化对风速有什么影响？',
                'answer': '气压梯度是风的直接驱动力。当两地气压差形成气压梯度时，空气从高压区流向低压区产生风。气压梯度越大，风速越高。同时，高气压本身也增加空气密度，一个1020hPa的高压区比1000hPa的低压区空气密度高约2%，有利于风能转换。'
            },
            {
                'question': '为什么风机有切入风速和切出风速？',
                'answer': '切入风速（约3m/s）是风机启动的最低风速，低于此值时风能不足以克服机械摩擦和发电机负载。切出风速（约25m/s）是保护性停机风速，超过此值时，过大的风力会对叶片、塔架和传动系统造成结构性损坏，因此必须制动停机。在3-25m/s之间是风机的运行区间。'
            },
            {
                'question': '空气密度为什么影响风电功率？',
                'answer': '空气密度决定了单位体积空气的质量，进而影响风能大小。根据动能公式E = 0.5 × m × v²，质量m越大，动能越大。空气密度每增加10%，在相同风速下功率也增加10%。这就是为什么高海拔地区（空气稀薄）风电效率低于平原地区的原因。'
            },
            {
                'question': '季风如何影响风电预测？',
                'answer': '季风是大规模的季节性风向和风速变化规律。夏季风通常来自海洋，温暖潮湿但风速较弱；冬季风来自大陆，寒冷干燥且风速强劲。对于风电预测，季风规律提供了重要的先验知识：冬季风期应预期更高的发电量，夏季风期则需降低预期。准确把握季风转换时机可将预测误差降低15-20%。'
            },
            {
                'question': '为什么风速波动会影响电网稳定？',
                'answer': '风电功率与风速三次方成正比，这意味着风速的小幅波动会导致功率大幅变化。例如，风速从10m/s波动到9m/s（降低10%），功率将下降约27%（0.9³≈0.73）。频繁的大幅功率波动给电网调度带来挑战，需要配合储能系统或火电调峰来平滑输出。'
            },
            {
                'question': '叶片效率（功率系数Cp）的物理意义是什么？',
                'answer': '功率系数Cp表示风机将风能转化为机械能的效率。根据贝茨定律，理论最大Cp为0.593（59.3%），实际风机Cp通常在0.35-0.45之间。这意味着即使在理想条件下，风机也只能捕获不到60%的风能，其余能量随尾流散失。提高Cp是风机设计的核心目标。'
            },
            {
                'question': '为什么高原地区风电效率较低？',
                'answer': '高原地区效率低主要是空气密度问题。海拔每升高1000米，气压下降约10%，空气密度也相应降低。例如，海拔3000米的高原空气密度约为平原的70%，这直接导致相同风速下的功率降低30%。虽然高原风速通常较大，但难以完全弥补密度损失。'
            }
        ]
        
        # 随机采样
        selected = random.sample(causality_pairs, min(n_samples, len(causality_pairs)))
        
        instructions = []
        for pair in selected:
            instructions.append({
                'instruction': pair['question'],
                'input': '',
                'output': pair['answer'],
                'type': 'causality'
            })
        
        print(f"✅ 生成{len(instructions)}条因果推理数据")
        return instructions
    
    def generate_type3_anomaly(self, n_samples: int = 30) -> List[Dict]:
        """
        类型3：异常分析类指令数据
        
        参数:
            n_samples: 生成样本数
            
        返回:
            指令数据列表
        """
        print(f"\n📝 生成类型3：异常分析类指令数据 ({n_samples}条)...")
        
        # 预定义异常场景
        anomaly_scenarios = [
            {
                'question': '风速突然下降但功率未立即下降，可能原因是什么？',
                'answer': '可能存在以下原因：1）气流惯性：风轮转速具有惯性，风速下降后需要一定时间才能减速；2）数据延迟：风速传感器和功率计量装置可能存在时间差；3）变桨系统响应滞后：叶片角度调整需要时间；4）多台风机平均效应：风场统计功率平滑了瞬时波动。建议检查数据时间戳和传感器同步性。'
            },
            {
                'question': '为什么同样风速下，不同季节的功率有差异？',
                'answer': '主要原因是空气密度的季节性变化。冬季低温使空气密度比夏季高10-15%，相同风速下冬季功率更高。例如，10m/s风速在冬季（密度1.30kg/m³）产生的功率比夏季（密度1.15kg/m³）高约13%。此外，季节性气压系统也影响风速稳定性，冬季风况通常更稳定。'
            },
            {
                'question': '预测功率远高于理论功率，应如何排查？',
                'answer': '排查步骤：1）检查输入数据：风速、密度等是否异常高值；2）验证模型输出：是否超出物理上限（如超过额定功率）；3）检查数据单位：功率单位是否统一（kW vs MW）；4）核对时间对齐：预测时刻与输入数据是否匹配；5）模型过拟合：训练集上表现好但泛化差。建议先进行物理合理性检查，再调整模型。'
            },
            {
                'question': '风速正常但功率异常低，可能是什么问题？',
                'answer': '可能问题包括：1）设备故障：叶片结冰、变桨卡滞、发电机故障；2）限电停机：电网调度要求降载或停机；3）维护状态：风机处于检修或调试状态；4）传感器故障：功率计量装置失准；5）尾流效应：上游风机遮挡导致实际风速低于测量值。需结合SCADA系统状态码和维护记录综合分析。'
            },
            {
                'question': '为什么夜间功率预测误差通常较大？',
                'answer': '夜间误差大的原因：1）边界层变化：夜间大气稳定度增加，风切变增强，测风塔高度的风速不能代表轮毂高度；2）低层急流：夜间可能形成低空急流，导致风速突变；3）预报模式缺陷：数值天气预报对夜间边界层模拟精度较低；4）历史数据稀疏：夜间训练样本可能不足。建议夜间预测时增加不确定性估计范围。'
            },
            {
                'question': '功率曲线出现多个平台段，如何解释？',
                'answer': '平台段对应风机的不同运行模式：1）3-8m/s：低速运行段，功率缓慢增长；2）8-12m/s：最优运行段，功率快速增长；3）12-15m/s：接近额定功率，增长减缓；4）15-25m/s：恒功率运行段，变桨控制维持额定功率；5）>25m/s：停机保护，功率为零。多平台是风机主动控制的结果，反映了不同风速下的优化策略。'
            },
            {
                'question': '为什么春季功率预测不确定性最大？',
                'answer': '春季预测难度大的原因：1）季节转换期：冬季风和夏季风交替，风况不稳定；2）天气系统频繁：冷暖气团交锋导致天气突变；3）温度波动大：昼夜温差可达15-20℃，空气密度变化显著；4）地形影响复杂：春季大气不稳定加剧地形对气流的扰动作用。建议春季预测时缩短预测时间跨度，增加更新频率。'
            },
            {
                'question': '密度计算值与实测值差异较大，如何处理？',
                'answer': '差异来源可能是：1）测量误差：温度、气压传感器精度不足；2）湿度影响：理想气体公式未考虑水汽，高湿度时实际密度低于计算值；3）海拔修正：气压需根据测量高度修正；4）数据时空不匹配：温度和气压测量位置、时间不一致。处理方法：优先使用实测密度，若无实测则用修正公式ρ = P/(R×T×(1+0.378×e/P))，其中e是水汽压。'
            },
            {
                'question': '为什么台风天气风速很大但功率为零？',
                'answer': '台风天气风速通常超过25m/s，达到风机切出风速阈值。此时风机会自动停机保护，原因：1）叶片强度限制：过大风力会导致叶片结构破坏；2）塔架安全：极端风载可能导致塔架倾覆；3）传动系统：齿轮箱、轴承无法承受过载；4）电气保护：发电机过流保护触发。宁可损失发电量也要保证设备安全，是风电运维的基本原则。'
            },
            {
                'question': '相邻风机功率差异明显，如何分析？',
                'answer': '差异原因分析：1）尾流效应：上风向风机遮挡下风向风机，导致后者功率降低20-40%；2）地形影响：山丘、树林等障碍物改变局部风况；3）设备差异：不同风机型号、叶片磨损程度不同；4）控制策略：可能实施了分组控制或限电策略；5）数据错误：传感器故障或通信异常。建议结合风向、风机布局和SCADA日志综合判断。'
            }
        ]
        
        # 随机采样
        selected = random.sample(anomaly_scenarios, min(n_samples, len(anomaly_scenarios)))
        
        instructions = []
        for scenario in selected:
            instructions.append({
                'instruction': scenario['question'],
                'input': '',
                'output': scenario['answer'],
                'type': 'anomaly'
            })
        
        print(f"✅ 生成{len(instructions)}条异常分析数据")
        return instructions
    
    def generate_all(self, output_file: str = 'wind_power_instruction_dataset.json'):
        """
        生成完整指令数据集
        
        参数:
            output_file: 输出文件路径
        """
        print("\n" + "="*80)
        print("🌟 开始生成风电领域指令微调数据集")
        print("="*80)
        
        all_instructions = []
        
        # 生成三类数据
        all_instructions.extend(self.generate_type1_explanation(n_samples=100))
        all_instructions.extend(self.generate_type2_causality(n_samples=50))
        all_instructions.extend(self.generate_type3_anomaly(n_samples=30))
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_instructions, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*80)
        print(f"✅ 数据集生成完成！")
        print(f"   总计：{len(all_instructions)}条指令数据")
        print(f"   - 解释类：{sum(1 for x in all_instructions if x['type']=='explanation')}条")
        print(f"   - 因果推理类：{sum(1 for x in all_instructions if x['type']=='causality')}条")
        print(f"   - 异常分析类：{sum(1 for x in all_instructions if x['type']=='anomaly')}条")
        print(f"   保存至：{output_file}")
        print("="*80)
        
        # 显示示例
        print("\n📖 数据集示例：")
        print("-"*80)
        for example in all_instructions[:3]:
            print(f"\n【{example['type']}】")
            print(f"Instruction: {example['instruction']}")
            if example['input']:
                print(f"Input: {example['input'][:100]}...")
            print(f"Output: {example['output'][:150]}...")
            print("-"*80)
        
        return all_instructions


def main():
    """主函数"""
    
    print("\n" + "🎯"*30)
    print("     风电领域指令微调数据集生成器")
    print("🎯"*30 + "\n")
    
    # 初始化生成器
    generator = WindPowerInstructionGenerator()
    
    # 生成数据集
    instructions = generator.generate_all()
    
    print("\n💡 下一步：")
    print("   1. 使用LoRA微调Qwen/LLaMA模型")
    print("   2. 训练脚本参考：")
    print("      - 模型：Qwen2.5-7B-Instruct")
    print("      - 方法：LoRA (r=8, alpha=16)")
    print("      - 数据格式：Alpaca格式（instruction+input+output）")
    print("   3. 微调后模型将具备风电领域专业解释能力")
    
    return instructions


if __name__ == "__main__":
    instructions = main()
