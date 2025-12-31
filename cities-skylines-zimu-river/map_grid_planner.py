"""
Cities: Skylines - 子母河地图网格规划工具
用于初始城市规划的网格化分析和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class GridCell:
    """网格单元数据结构"""
    x: int
    y: int
    terrain: str  # 'land', 'water', 'mountain', 'forest'
    zone_type: str  # 'residential', 'commercial', 'industrial', 'office', 'mixed', 'park', 'unplanned'
    priority: int  # 1-5, 5为最高优先级
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class ZimuRiverMapPlanner:
    """子母河地图规划器"""

    # 区域类型配色方案（参考Cities: Skylines的配色）
    ZONE_COLORS = {
        'residential': '#4CAF50',    # 绿色 - 住宅
        'commercial': '#2196F3',     # 蓝色 - 商业
        'industrial': '#FFC107',     # 黄色 - 工业
        'office': '#00BCD4',         # 青色 - 办公
        'mixed': '#9C27B0',          # 紫色 - 混合用途
        'park': '#8BC34A',           # 浅绿 - 公园绿地
        'unplanned': '#E0E0E0'       # 灰色 - 未规划
    }

    TERRAIN_COLORS = {
        'land': '#D7CCC8',           # 棕色 - 陆地
        'water': '#64B5F6',          # 浅蓝 - 水域
        'mountain': '#795548',       # 深棕 - 山地
        'forest': '#689F38'          # 深绿 - 森林
    }

    def __init__(self, width: int = 30, height: int = 30):
        """
        初始化地图规划器

        Args:
            width: 网格宽度（默认30格，代表约3km）
            height: 网格高度（默认30格）
        """
        self.width = width
        self.height = height
        self.grid: List[List[GridCell]] = []
        self._initialize_grid()

    def _initialize_grid(self):
        """初始化网格，所有单元默认为未规划的陆地"""
        for y in range(self.height):
            row = []
            for x in range(self.width):
                cell = GridCell(
                    x=x,
                    y=y,
                    terrain='land',
                    zone_type='unplanned',
                    priority=1
                )
                row.append(cell)
            self.grid.append(row)

    def set_terrain(self, x: int, y: int, terrain: str):
        """设置地形类型"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].terrain = terrain

    def set_zone(self, x: int, y: int, zone_type: str, priority: int = 3, notes: str = ""):
        """设置区域规划"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].zone_type = zone_type
            self.grid[y][x].priority = priority
            self.grid[y][x].notes = notes

    def set_area(self, x_start: int, y_start: int, x_end: int, y_end: int,
                 zone_type: str, priority: int = 3, notes: str = ""):
        """批量设置区域"""
        for y in range(y_start, min(y_end + 1, self.height)):
            for x in range(x_start, min(x_end + 1, self.width)):
                self.set_zone(x, y, zone_type, priority, notes)

    def set_river(self, path: List[Tuple[int, int]], width: int = 1):
        """设置河流路径"""
        for x, y in path:
            for dy in range(-width // 2, width // 2 + 1):
                for dx in range(-width // 2, width // 2 + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.set_terrain(nx, ny, 'water')

    def visualize(self, show_terrain: bool = True, show_zones: bool = True,
                  show_grid: bool = True, show_priority: bool = False):
        """
        可视化地图规划

        Args:
            show_terrain: 显示地形层
            show_zones: 显示区域规划层
            show_grid: 显示网格线
            show_priority: 显示优先级数字
        """
        fig, axes = plt.subplots(1, 2 if show_terrain and show_zones else 1,
                                 figsize=(16, 8))

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        plot_idx = 0

        # 绘制地形图
        if show_terrain:
            ax_terrain = axes[plot_idx]
            terrain_grid = np.zeros((self.height, self.width))
            terrain_map = {'land': 0, 'water': 1, 'mountain': 2, 'forest': 3}

            for y in range(self.height):
                for x in range(self.width):
                    terrain_grid[y, x] = terrain_map.get(self.grid[y][x].terrain, 0)

            colors = [self.TERRAIN_COLORS[t] for t in ['land', 'water', 'mountain', 'forest']]
            cmap = ListedColormap(colors)

            ax_terrain.imshow(terrain_grid, cmap=cmap, interpolation='nearest', origin='upper')
            ax_terrain.set_title('地形分布图 (Terrain Map)', fontsize=14, fontweight='bold', pad=20)
            ax_terrain.set_xlabel('X 坐标 (East →)', fontsize=11)
            ax_terrain.set_ylabel('Y 坐标 (South ↓)', fontsize=11)

            if show_grid:
                for x in range(self.width + 1):
                    ax_terrain.axvline(x - 0.5, color='white', linewidth=0.5, alpha=0.3)
                for y in range(self.height + 1):
                    ax_terrain.axhline(y - 0.5, color='white', linewidth=0.5, alpha=0.3)

            # 添加图例
            legend_elements = [patches.Patch(facecolor=self.TERRAIN_COLORS[t], label=t.title())
                             for t in ['land', 'water', 'mountain', 'forest']]
            ax_terrain.legend(handles=legend_elements, loc='upper left', fontsize=9)

            plot_idx += 1

        # 绘制区域规划图
        if show_zones:
            ax_zones = axes[plot_idx]
            zone_grid = np.zeros((self.height, self.width))
            zone_list = list(self.ZONE_COLORS.keys())
            zone_map = {z: i for i, z in enumerate(zone_list)}

            for y in range(self.height):
                for x in range(self.width):
                    zone_grid[y, x] = zone_map.get(self.grid[y][x].zone_type, 0)

            colors = [self.ZONE_COLORS[z] for z in zone_list]
            cmap = ListedColormap(colors)

            ax_zones.imshow(zone_grid, cmap=cmap, interpolation='nearest', origin='upper')
            ax_zones.set_title('区域规划图 (Zoning Plan)', fontsize=14, fontweight='bold', pad=20)
            ax_zones.set_xlabel('X 坐标 (East →)', fontsize=11)
            ax_zones.set_ylabel('Y 坐标 (South ↓)', fontsize=11)

            if show_grid:
                for x in range(self.width + 1):
                    ax_zones.axvline(x - 0.5, color='white', linewidth=0.5, alpha=0.3)
                for y in range(self.height + 1):
                    ax_zones.axhline(y - 0.5, color='white', linewidth=0.5, alpha=0.3)

            # 显示优先级
            if show_priority:
                for y in range(self.height):
                    for x in range(self.width):
                        priority = self.grid[y][x].priority
                        if priority > 1:
                            ax_zones.text(x, y, str(priority), ha='center', va='center',
                                        color='white', fontsize=8, fontweight='bold')

            # 添加图例
            legend_elements = [patches.Patch(facecolor=self.ZONE_COLORS[z],
                                           label=z.replace('_', ' ').title())
                             for z in zone_list]
            ax_zones.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)

        plt.tight_layout()
        return fig

    def get_statistics(self) -> Dict:
        """获取规划统计信息"""
        stats = {
            'terrain': {},
            'zones': {},
            'total_cells': self.width * self.height
        }

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                stats['terrain'][cell.terrain] = stats['terrain'].get(cell.terrain, 0) + 1
                stats['zones'][cell.zone_type] = stats['zones'].get(cell.zone_type, 0) + 1

        # 计算百分比
        for key in stats['terrain']:
            stats['terrain'][key] = {
                'count': stats['terrain'][key],
                'percentage': round(stats['terrain'][key] / stats['total_cells'] * 100, 2)
            }

        for key in stats['zones']:
            stats['zones'][key] = {
                'count': stats['zones'][key],
                'percentage': round(stats['zones'][key] / stats['total_cells'] * 100, 2)
            }

        return stats

    def save_to_json(self, filename: str):
        """保存规划到JSON文件"""
        data = {
            'width': self.width,
            'height': self.height,
            'grid': [[cell.to_dict() for cell in row] for row in self.grid],
            'statistics': self.get_statistics()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print("子母河地图规划统计 (Zimu River Map Planning Statistics)")
        print("="*60)

        print(f"\n网格尺寸: {self.width} × {self.height} = {stats['total_cells']} 单元格")

        print("\n【地形分布】")
        for terrain, data in sorted(stats['terrain'].items()):
            print(f"  {terrain:12s}: {data['count']:4d} 格 ({data['percentage']:5.2f}%)")

        print("\n【区域规划】")
        for zone, data in sorted(stats['zones'].items()):
            print(f"  {zone:12s}: {data['count']:4d} 格 ({data['percentage']:5.2f}%)")


def create_zimu_river_initial_plan():
    """
    创建子母河地图的初始规划

    子母河地图特点：
    - 中央有一条主要河流（母河）穿过
    - 有支流（子河）汇入
    - 地形相对平坦，适合建设
    - 需要考虑水资源管理和防洪
    """

    planner = ZimuRiverMapPlanner(width=30, height=30)

    # ========== 第1阶段：设置地形 ==========

    # 绘制主河道（母河）- 从西北到东南的对角线河流
    main_river_path = []
    for i in range(35):
        x = int(i * 0.8)
        y = int(5 + i * 0.7)
        if x < 30 and y < 30:
            main_river_path.append((x, y))

    planner.set_river(main_river_path, width=2)

    # 绘制支流（子河）- 从东北汇入主河
    sub_river_path = []
    for i in range(15):
        x = 20 + i // 3
        y = 5 + i
        if x < 30 and y < 30:
            sub_river_path.append((x, y))

    planner.set_river(sub_river_path, width=1)

    # 设置一些森林区域（河流附近的绿化带）
    planner.set_area(0, 0, 5, 5, zone_type='unplanned', priority=1)
    for y in range(5):
        for x in range(6):
            planner.set_terrain(x, y, 'forest')

    # ========== 第2阶段：区域规划 ==========

    # 起始区域（左上角）- 住宅为主
    planner.set_area(0, 6, 7, 12, 'residential', priority=5,
                     notes="起始住宅区 - 靠近水源，环境优美")

    # 初期商业区（起始区旁边）
    planner.set_area(8, 6, 11, 10, 'commercial', priority=5,
                     notes="初期商业中心 - 服务起始住宅区")

    # 轻工业区（河流下游，远离住宅）
    planner.set_area(5, 20, 10, 25, 'industrial', priority=4,
                     notes="轻工业区 - 靠近河流，运输便利")

    # 中期扩展住宅区（右上角）
    planner.set_area(20, 0, 29, 8, 'residential', priority=3,
                     notes="中期住宅区 - 地势较高，视野好")

    # 办公区（城市中心）
    planner.set_area(12, 12, 18, 16, 'office', priority=3,
                     notes="CBD核心区 - 交通枢纽位置")

    # 混合用途区（办公区周边）
    planner.set_area(12, 17, 18, 20, 'mixed', priority=3,
                     notes="城市副中心 - 商住混合")

    # 公园绿地（河流两岸）
    planner.set_area(2, 13, 4, 18, 'park', priority=4,
                     notes="滨河公园 - 提升居住品质")

    # 后期发展住宅区（右下角）
    planner.set_area(20, 22, 29, 29, 'residential', priority=2,
                     notes="后期扩展住宅区 - 需要完善基础设施")

    # 商业次中心（右侧中部）
    planner.set_area(24, 14, 29, 18, 'commercial', priority=2,
                     notes="商业次中心 - 服务东部住宅")

    return planner


if __name__ == "__main__":
    print("正在生成子母河地图初始规划...")

    # 创建规划
    planner = create_zimu_river_initial_plan()

    # 打印统计信息
    planner.print_statistics()

    # 保存到JSON
    planner.save_to_json('zimu_river_plan.json')
    print("\n✓ 规划已保存到 zimu_river_plan.json")

    # 生成可视化
    fig = planner.visualize(show_terrain=True, show_zones=True,
                           show_grid=True, show_priority=True)

    plt.savefig('zimu_river_plan.png', dpi=300, bbox_inches='tight')
    print("✓ 可视化图表已保存到 zimu_river_plan.png")

    plt.show()

    print("\n" + "="*60)
    print("规划建议：")
    print("="*60)
    print("""
    【初期发展建议】（0-5万人口）
    1. 从左上角起始住宅区开始建设
    2. 配套建设初期商业区提供服务
    3. 优先建设供水、供电和道路基础设施
    4. 沿主河道建设取水设施
    5. 规划主干道连接各功能区

    【中期发展建议】（5-15万人口）
    6. 开发右上角住宅区，形成双中心格局
    7. 建设CBD办公区，发展服务业
    8. 完善轻工业区，注意污染控制
    9. 加强公共交通建设（地铁/公交）
    10. 建设滨河公园，提升城市品质

    【后期发展建议】（15万+人口）
    11. 扩展右下角住宅区，完善配套设施
    12. 发展商业次中心，分散城市压力
    13. 升级工业区为高科技园区
    14. 完善教育、医疗等公共服务设施
    15. 优化交通网络，解决拥堵问题

    【关键注意事项】
    - 河流管理：注意上游取水，下游排污的合理布局
    - 交通规划：预留主干道和高速公路接口位置
    - 防洪措施：河流两岸保留足够的防洪空间
    - 环境保护：保留森林区，控制工业污染
    - 分期建设：按优先级（priority）依次开发
    """)
