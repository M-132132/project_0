"""
轨迹预测归因结果分析脚本

提供深入的归因结果分析，包括统计分析、对比分析、模式发现等

使用方法:
    python exps_scripts/exp_trajattr/analyze_attr_results.py --result_path exps_res/res_trajattr/autobot_nuscenes/20240101_120000
    python exps_scripts/exp_trajattr/analyze_attr_results.py --batch_result_path exps_res/res_trajattr/batch_experiments/batch_report_20240101_120000.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 添加项目根路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class TrajAttrAnalyzer:
    """轨迹预测归因结果分析器"""
    
    def __init__(self, result_path: str = None, batch_result_path: str = None):
        self.result_path = Path(result_path) if result_path else None
        self.batch_result_path = Path(batch_result_path) if batch_result_path else None
        
        # 分析配置
        self.analysis_config = {
            'statistical_tests': ['shapiro', 'kstest', 'ttest'],
            'clustering_methods': ['kmeans', 'hierarchical'],
            'dimensionality_reduction': ['pca', 'tsne'],
            'significance_level': 0.05,
            'n_clusters': 5
        }
        
        # 数据存储
        self.single_exp_data = None
        self.batch_exp_data = None
        self.analysis_results = {}
        
    def load_data(self):
        """加载分析数据"""
        if self.result_path and self.result_path.exists():
            self.single_exp_data = self._load_single_experiment_data()
            
        if self.batch_result_path and self.batch_result_path.exists():
            self.batch_exp_data = self._load_batch_experiment_data()
            
    def _load_single_experiment_data(self) -> Dict:
        """加载单个实验数据"""
        print(f"加载单个实验数据: {self.result_path}")
        
        data = {
            'attributions': {},
            'metadata': {},
            'statistics': {}
        }
        
        # 加载归因数据
        numpy_dir = self.result_path / 'attributions' / 'numpy'
        if numpy_dir.exists():
            attribution_files = {}
            for npy_file in numpy_dir.glob('*.npy'):
                filename_parts = npy_file.stem.split('_')
                if len(filename_parts) >= 4:
                    batch_id = f"{filename_parts[0]}_{filename_parts[1]}"
                    method = filename_parts[2]
                    input_type = '_'.join(filename_parts[3:])
                    
                    if batch_id not in attribution_files:
                        attribution_files[batch_id] = {}
                    if method not in attribution_files[batch_id]:
                        attribution_files[batch_id][method] = {}
                    
                    attribution_files[batch_id][method][input_type] = np.load(npy_file)
            
            data['attributions'] = attribution_files
        
        # 加载实验报告
        report_path = self.result_path / 'reports' / 'experiment_report.json'
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                data['metadata'] = json.load(f)
        
        # 加载分析结果
        analysis_path = self.result_path / 'reports' / 'attribution_analysis.json'
        if analysis_path.exists():
            with open(analysis_path, 'r', encoding='utf-8') as f:
                data['statistics'] = json.load(f)
        
        return data
    
    def _load_batch_experiment_data(self) -> Dict:
        """加载批量实验数据"""
        print(f"加载批量实验数据: {self.batch_result_path}")
        
        with open(self.batch_result_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_comprehensive_analysis(self) -> Dict:
        """运行综合分析"""
        print("="*60)
        print("开始综合归因分析")
        print("="*60)
        
        # 加载数据
        self.load_data()
        
        analysis_results = {}
        
        # 单实验分析
        if self.single_exp_data:
            print("\\n进行单实验深度分析...")
            analysis_results['single_experiment'] = self._analyze_single_experiment()
        
        # 批量实验分析
        if self.batch_exp_data:
            print("\\n进行批量实验对比分析...")
            analysis_results['batch_experiments'] = self._analyze_batch_experiments()
        
        # 跨实验分析（如果两种数据都存在）
        if self.single_exp_data and self.batch_exp_data:
            print("\\n进行跨实验综合分析...")
            analysis_results['cross_experiment'] = self._analyze_cross_experiments()
        
        self.analysis_results = analysis_results
        
        # 生成分析报告
        self._generate_analysis_report()
        
        print("\\n" + "="*60)
        print("综合分析完成！")
        print("="*60)
        
        return analysis_results
    
    def _analyze_single_experiment(self) -> Dict:
        """分析单个实验"""
        results = {
            'statistical_analysis': {},
            'distribution_analysis': {},
            'correlation_analysis': {},
            'clustering_analysis': {},
            'importance_ranking': {}
        }
        
        attributions = self.single_exp_data['attributions']
        
        # 统计分析
        results['statistical_analysis'] = self._perform_statistical_analysis(attributions)
        
        # 分布分析
        results['distribution_analysis'] = self._analyze_distributions(attributions)
        
        # 相关性分析
        results['correlation_analysis'] = self._analyze_correlations(attributions)
        
        # 聚类分析
        results['clustering_analysis'] = self._perform_clustering_analysis(attributions)
        
        # 重要性排名
        results['importance_ranking'] = self._rank_feature_importance(attributions)
        
        return results
    
    def _perform_statistical_analysis(self, attributions: Dict) -> Dict:
        """执行统计分析"""
        stats_results = {}
        
        for batch_id, methods in attributions.items():
            stats_results[batch_id] = {}
            
            for method, inputs in methods.items():
                method_stats = {}
                
                for input_type, data in inputs.items():
                    # 扁平化数据
                    flat_data = data.flatten()
                    
                    # 基础统计
                    basic_stats = {
                        'mean': float(np.mean(flat_data)),
                        'std': float(np.std(flat_data)),
                        'median': float(np.median(flat_data)),
                        'min': float(np.min(flat_data)),
                        'max': float(np.max(flat_data)),
                        'q25': float(np.percentile(flat_data, 25)),
                        'q75': float(np.percentile(flat_data, 75)),
                        'skewness': float(stats.skew(flat_data)),
                        'kurtosis': float(stats.kurtosis(flat_data))
                    }
                    
                    # 正态性检验
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(flat_data[:5000])  # 限制样本数
                        basic_stats['normality_test'] = {
                            'shapiro_stat': float(shapiro_stat),
                            'shapiro_p': float(shapiro_p),
                            'is_normal': shapiro_p > self.analysis_config['significance_level']
                        }
                    except:
                        basic_stats['normality_test'] = {'error': 'Failed to perform test'}
                    
                    # 零值分析
                    zero_ratio = np.sum(np.abs(flat_data) < 1e-6) / len(flat_data)
                    basic_stats['sparsity'] = {
                        'zero_ratio': float(zero_ratio),
                        'non_zero_count': int(np.sum(np.abs(flat_data) >= 1e-6))
                    }
                    
                    method_stats[input_type] = basic_stats
                
                stats_results[batch_id][method] = method_stats
        
        return stats_results
    
    def _analyze_distributions(self, attributions: Dict) -> Dict:
        """分析归因值分布"""
        distribution_results = {}
        
        # 收集所有归因值
        all_values_by_method = {}
        
        for batch_id, methods in attributions.items():
            for method, inputs in methods.items():
                if method not in all_values_by_method:
                    all_values_by_method[method] = []
                
                for input_type, data in inputs.items():
                    all_values_by_method[method].extend(data.flatten())
        
        # 分析每种方法的分布
        for method, values in all_values_by_method.items():
            values = np.array(values)
            
            # 分位数分析
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            percentile_values = {f'p{p}': float(np.percentile(values, p)) for p in percentiles}
            
            # 异常值检测
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            distribution_results[method] = {
                'percentiles': percentile_values,
                'outlier_analysis': {
                    'outlier_count': int(len(outliers)),
                    'outlier_ratio': float(len(outliers) / len(values)),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                },
                'distribution_shape': {
                    'skewness': float(stats.skew(values)),
                    'kurtosis': float(stats.kurtosis(values)),
                    'range': float(values.max() - values.min())
                }
            }
        
        return distribution_results
    
    def _analyze_correlations(self, attributions: Dict) -> Dict:
        """分析不同方法间的相关性"""
        correlation_results = {}
        
        # 收集数据用于相关性分析
        method_data = {}
        
        for batch_id, methods in attributions.items():
            for method, inputs in methods.items():
                if method not in method_data:
                    method_data[method] = []
                
                # 将所有输入类型的数据合并
                batch_data = []
                for input_type, data in inputs.items():
                    batch_data.extend(data.flatten())
                
                method_data[method].append(batch_data)
        
        # 确保所有方法有相同数量的批次
        methods_list = list(method_data.keys())
        if len(methods_list) > 1:
            min_batches = min(len(data) for data in method_data.values())
            
            # 计算两两相关性
            correlations = {}
            for i, method1 in enumerate(methods_list):
                for j, method2 in enumerate(methods_list[i+1:], i+1):
                    corr_values = []
                    
                    for batch_idx in range(min_batches):
                        data1 = np.array(method_data[method1][batch_idx])
                        data2 = np.array(method_data[method2][batch_idx])
                        
                        # 确保数据长度相同
                        min_len = min(len(data1), len(data2))
                        data1 = data1[:min_len]
                        data2 = data2[:min_len]
                        
                        try:
                            corr = np.corrcoef(data1, data2)[0, 1]
                            if not np.isnan(corr):
                                corr_values.append(corr)
                        except:
                            continue
                    
                    if corr_values:
                        correlations[f'{method1}_vs_{method2}'] = {
                            'mean_correlation': float(np.mean(corr_values)),
                            'std_correlation': float(np.std(corr_values)),
                            'correlations': [float(c) for c in corr_values]
                        }
            
            correlation_results['pairwise_correlations'] = correlations
        
        return correlation_results
    
    def _perform_clustering_analysis(self, attributions: Dict) -> Dict:
        """执行聚类分析"""
        clustering_results = {}
        
        # 为每种方法执行聚类
        for batch_id, methods in attributions.items():
            clustering_results[batch_id] = {}
            
            for method, inputs in methods.items():
                method_clustering = {}
                
                for input_type, data in inputs.items():
                    if input_type == 'obj_trajs':  # 只对轨迹数据进行聚类
                        # 重塑数据为2D
                        original_shape = data.shape
                        if len(original_shape) == 3:  # [N, T, F]
                            reshaped_data = data.reshape(original_shape[0], -1)  # [N, T*F]
                        else:
                            continue
                        
                        # K-means聚类
                        try:
                            n_clusters = min(self.analysis_config['n_clusters'], reshaped_data.shape[0])
                            if n_clusters > 1:
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                cluster_labels = kmeans.fit_predict(reshaped_data)
                                
                                # 计算轮廓系数
                                from sklearn.metrics import silhouette_score
                                silhouette = silhouette_score(reshaped_data, cluster_labels)
                                
                                method_clustering[input_type] = {
                                    'n_clusters': n_clusters,
                                    'cluster_labels': cluster_labels.tolist(),
                                    'cluster_centers': kmeans.cluster_centers_.tolist(),
                                    'silhouette_score': float(silhouette),
                                    'inertia': float(kmeans.inertia_)
                                }
                        except Exception as e:
                            method_clustering[input_type] = {'error': str(e)}
                
                clustering_results[batch_id][method] = method_clustering
        
        return clustering_results
    
    def _rank_feature_importance(self, attributions: Dict) -> Dict:
        """排名特征重要性"""
        importance_rankings = {}
        
        # 特征类型定义
        feature_names = {
            'obj_trajs': ['pos_x', 'pos_y', 'vel_x', 'vel_y', 'acc_x', 'acc_y', 'heading', 'others'],
            'map_polylines': ['map_pos_x', 'map_pos_y', 'map_attr']
        }
        
        for batch_id, methods in attributions.items():
            importance_rankings[batch_id] = {}
            
            for method, inputs in methods.items():
                method_rankings = {}
                
                for input_type, data in inputs.items():
                    if input_type in feature_names:
                        # 计算每个特征维度的重要性
                        if len(data.shape) >= 3:
                            # 对前几个维度求平均，保留特征维度
                            feature_importance = np.abs(data).mean(axis=tuple(range(len(data.shape)-1)))
                        else:
                            feature_importance = np.abs(data).mean(axis=0)
                        
                        # 排序并创建排名
                        sorted_indices = np.argsort(feature_importance)[::-1]
                        
                        rankings = []
                        names = feature_names[input_type]
                        for i, idx in enumerate(sorted_indices):
                            if idx < len(names):
                                rankings.append({
                                    'rank': i + 1,
                                    'feature': names[idx],
                                    'importance': float(feature_importance[idx]),
                                    'relative_importance': float(feature_importance[idx] / feature_importance.max())
                                })
                        
                        method_rankings[input_type] = rankings
                
                importance_rankings[batch_id][method] = method_rankings
        
        return importance_rankings
    
    def _analyze_batch_experiments(self) -> Dict:
        """分析批量实验结果"""
        results = {
            'performance_analysis': {},
            'method_comparison': {},
            'model_comparison': {},
            'success_rate_analysis': {}
        }
        
        detailed_results = self.batch_exp_data.get('detailed_results', [])
        
        # 性能分析
        durations = [r['duration'] for r in detailed_results if 'duration' in r]
        success_rates = [r['success'] for r in detailed_results]
        
        results['performance_analysis'] = {
            'average_duration': float(np.mean(durations)) if durations else 0,
            'total_duration': float(np.sum(durations)) if durations else 0,
            'success_rate': float(np.mean(success_rates)) if success_rates else 0,
            'total_experiments': len(detailed_results)
        }
        
        # 按方法分组分析
        method_performance = {}
        model_performance = {}
        
        for result in detailed_results:
            exp_config = result.get('exp_config', {})
            methods = exp_config.get('methods', [])
            model = exp_config.get('model', 'unknown')
            
            # 方法性能
            method_key = '_'.join(sorted(methods))
            if method_key not in method_performance:
                method_performance[method_key] = {'successes': 0, 'total': 0, 'durations': []}
            
            method_performance[method_key]['total'] += 1
            if result.get('success', False):
                method_performance[method_key]['successes'] += 1
            if 'duration' in result:
                method_performance[method_key]['durations'].append(result['duration'])
            
            # 模型性能
            if model not in model_performance:
                model_performance[model] = {'successes': 0, 'total': 0, 'durations': []}
            
            model_performance[model]['total'] += 1
            if result.get('success', False):
                model_performance[model]['successes'] += 1
            if 'duration' in result:
                model_performance[model]['durations'].append(result['duration'])
        
        # 计算汇总统计
        for method_key, stats in method_performance.items():
            stats['success_rate'] = stats['successes'] / stats['total'] if stats['total'] > 0 else 0
            stats['avg_duration'] = np.mean(stats['durations']) if stats['durations'] else 0
        
        for model, stats in model_performance.items():
            stats['success_rate'] = stats['successes'] / stats['total'] if stats['total'] > 0 else 0
            stats['avg_duration'] = np.mean(stats['durations']) if stats['durations'] else 0
        
        results['method_comparison'] = method_performance
        results['model_comparison'] = model_performance
        
        return results
    
    def _analyze_cross_experiments(self) -> Dict:
        """跨实验综合分析"""
        results = {
            'consistency_analysis': {},
            'scalability_analysis': {},
            'reliability_analysis': {}
        }
        
        # 这里可以添加更复杂的跨实验分析逻辑
        # 例如比较单实验和批量实验的一致性
        
        return results
    
    def _generate_analysis_report(self):
        """生成分析报告"""
        # 创建报告目录
        if self.result_path:
            report_dir = self.result_path / 'analysis_reports'
        else:
            report_dir = Path('exps_res/res_trajattr/analysis_reports')
        
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存完整分析结果
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        analysis_file = report_dir / f'comprehensive_analysis_{timestamp}.json'
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(self._convert_numpy_types(self.analysis_results), f, indent=2, ensure_ascii=False)
        
        # 生成可读报告
        self._generate_readable_report(report_dir, timestamp)
        
        print(f"\\n分析报告已保存到: {report_dir}")
    
    def _generate_readable_report(self, report_dir: Path, timestamp: str):
        """生成可读的分析报告"""
        report_lines = []
        report_lines.append("# 轨迹预测归因分析报告")
        report_lines.append(f"\\n生成时间: {timestamp}")
        report_lines.append("\\n" + "="*60)
        
        # 单实验分析报告
        if 'single_experiment' in self.analysis_results:
            single_exp = self.analysis_results['single_experiment']
            report_lines.append("\\n## 单实验深度分析")
            
            # 统计分析摘要
            if 'statistical_analysis' in single_exp:
                report_lines.append("\\n### 统计分析摘要")
                stats = single_exp['statistical_analysis']
                for batch_id, methods in stats.items():
                    report_lines.append(f"\\n**批次 {batch_id}:**")
                    for method, inputs in methods.items():
                        report_lines.append(f"  - {method}:")
                        for input_type, data in inputs.items():
                            mean_val = data.get('mean', 0)
                            std_val = data.get('std', 0)
                            report_lines.append(f"    - {input_type}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")
            
            # 分布分析摘要
            if 'distribution_analysis' in single_exp:
                report_lines.append("\\n### 分布分析摘要")
                dist = single_exp['distribution_analysis']
                for method, data in dist.items():
                    outlier_ratio = data['outlier_analysis']['outlier_ratio']
                    skewness = data['distribution_shape']['skewness']
                    report_lines.append(f"  - {method}: 异常值比例={outlier_ratio:.2%}, 偏度={skewness:.3f}")
        
        # 批量实验分析报告
        if 'batch_experiments' in self.analysis_results:
            batch_exp = self.analysis_results['batch_experiments']
            report_lines.append("\\n## 批量实验分析")
            
            if 'performance_analysis' in batch_exp:
                perf = batch_exp['performance_analysis']
                report_lines.append("\\n### 性能概览")
                report_lines.append(f"  - 总实验数: {perf['total_experiments']}")
                report_lines.append(f"  - 成功率: {perf['success_rate']:.2%}")
                report_lines.append(f"  - 平均耗时: {perf['average_duration']:.1f}秒")
                report_lines.append(f"  - 总耗时: {perf['total_duration']:.1f}秒")
            
            if 'method_comparison' in batch_exp:
                report_lines.append("\\n### 方法对比")
                method_comp = batch_exp['method_comparison']
                for method, stats in method_comp.items():
                    success_rate = stats['success_rate']
                    avg_duration = stats['avg_duration']
                    report_lines.append(f"  - {method}: 成功率={success_rate:.2%}, 平均耗时={avg_duration:.1f}s")
        
        # 保存可读报告
        readable_report_file = report_dir / f'analysis_summary_{timestamp}.md'
        with open(readable_report_file, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(report_lines))
        
    def _convert_numpy_types(self, obj):
        """转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='轨迹预测归因结果分析')
    parser.add_argument('--result_path', type=str, help='单个实验结果路径')
    parser.add_argument('--batch_result_path', type=str, help='批量实验结果文件路径')
    parser.add_argument('--analysis_types', nargs='+', 
                       choices=['statistical', 'distribution', 'correlation', 'clustering', 'ranking', 'all'],
                       default=['all'], help='要执行的分析类型')
    
    args = parser.parse_args()
    
    if not args.result_path and not args.batch_result_path:
        print("错误: 必须提供 --result_path 或 --batch_result_path")
        sys.exit(1)
    
    # 检查路径
    if args.result_path and not Path(args.result_path).exists():
        print(f"错误: 结果路径不存在 {args.result_path}")
        sys.exit(1)
        
    if args.batch_result_path and not Path(args.batch_result_path).exists():
        print(f"错误: 批量结果文件不存在 {args.batch_result_path}")
        sys.exit(1)
    
    try:
        # 创建分析器
        analyzer = TrajAttrAnalyzer(args.result_path, args.batch_result_path)
        
        # 运行分析
        results = analyzer.run_comprehensive_analysis()
        
        print(f"\\n✓ 分析完成！结果已保存")
        
    except Exception as e:
        print(f"\\n❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()