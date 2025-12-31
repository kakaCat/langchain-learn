"""
Cities: Skylines - 子母河地图 (Two Rivers Map) 网格规划工具
基于实际游戏地图的初始规划
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
    terrain: str  # 'land', 'water', 'highway', 'buildable', 'unbuildable'
    zone_type: str  # 'residential', 'commercial', 'industrial', 'office', 'mixed', 'park', 'unplanned'
    priority: int  # 1-5, 5为最高优先级
    phase: int  # 发展阶段 1-初期, 2-中期, 3-后期
    notes: str = ""

    def to_dict(self):
        return asdict(self)


class TwoRiversMapPlanner:
    """子母河地图规划器"""

    # 区域类型配色方案
    ZONE_COLORS = {
        'residential': '#4CAF50',    # 绿色 - 住宅
        'commercial': '#2196F3',     # 蓝色 - 商业
        'industrial': '#FFC107',     # 黄色 - 工业
        'office': '#00BCD4',         # 青色 - 办公
        'mixed': '#9C27B0',          # 紫色 - 混合用途
        'park': '#8BC34A',           # 浅绿 - 公园绿地
        'highway': '#424242',        # 深灰 - 高速公路
        'bridge': '#FF5722',         # 橙红 - 桥梁位置
        'unplanned': '#E0E0E0'       # 灰色 - 未规划
    }

    TERRAIN_COLORS = {
        'buildable': '#C8E6C9',      # 浅绿 - 可建设陆地
        'unbuildable': '#A1887F',    # 棕灰 - 不可建设
        'water': '#64B5F6',          # 蓝色 - 水域
        'highway': '#424242',        # 深灰 - 高速公路
        'forest': '#689F38'          # 深绿 - 森林
    }

    PHASE_COLORS = {
        1: '#FF5252',  # 红色 - 初期（高优先级）
        2: '#FFC107',  # 黄色 - 中期
        3: '#4CAF50',  # 绿色 - 后期
        0: '#E0E0E0'   # 灰色 - 未规划
    }

    def __init__(self, width: int = 40, height: int = 40):
        """
        初始化地图规划器

        基于子母河地图的实际尺寸：
        - 地图总大小约 2x2 瓦片 (2km x 2km)
        - 使用 40x40 网格，每格约 50m
        """
        self.width = width
        self.height = height
        self.grid: List[List[GridCell]] = []
        self.bridges: List[Tuple[int, int]] = []  # 桥梁位置
        self._initialize_grid()

    def _initialize_grid(self):
        """初始化网格"""
        for y in range(self.height):
            row = []
            for x in range(self.width):
                cell = GridCell(
                    x=x, y=y,
                    terrain='buildable',
                    zone_type='unplanned',
                    priority=1,
                    phase=0
                )
                row.append(cell)
            self.grid.append(row)

    def set_terrain(self, x: int, y: int, terrain: str):
        """设置地形类型"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].terrain = terrain

    def set_zone(self, x: int, y: int, zone_type: str, priority: int = 3,
                 phase: int = 1, notes: str = ""):
        """设置区域规划"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x].zone_type = zone_type
            self.grid[y][x].priority = priority
            self.grid[y][x].phase = phase
            self.grid[y][x].notes = notes

    def set_area(self, x_start: int, y_start: int, x_end: int, y_end: int,
                 zone_type: str, priority: int = 3, phase: int = 1, notes: str = ""):
        """批量设置区域"""
        for y in range(y_start, min(y_end + 1, self.height)):
            for x in range(x_start, min(x_end + 1, self.width)):
                if self.grid[y][x].terrain in ['buildable', 'forest']:
                    self.set_zone(x, y, zone_type, priority, phase, notes)

    def draw_river(self, path: List[Tuple[int, int]], width: int = 2):
        """绘制河流"""
        for x, y in path:
            for dy in range(-width // 2, width // 2 + 1):
                for dx in range(-width // 2, width // 2 + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.set_terrain(nx, ny, 'water')

    def add_bridge(self, x: int, y: int):
        """标记桥梁位置"""
        self.bridges.append((x, y))

    def visualize(self, show_phases: bool = True):
        """生成三视图：地形、区域规划、发展阶段"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # === 图1: 地形图 ===
        ax_terrain = axes[0]
        terrain_grid = np.zeros((self.height, self.width))
        terrain_types = ['buildable', 'water', 'unbuildable', 'highway', 'forest']
        terrain_map = {t: i for i, t in enumerate(terrain_types)}

        for y in range(self.height):
            for x in range(self.width):
                terrain_grid[y, x] = terrain_map.get(self.grid[y][x].terrain, 0)

        colors = [self.TERRAIN_COLORS[t] for t in terrain_types]
        cmap = ListedColormap(colors)

        ax_terrain.imshow(terrain_grid, cmap=cmap, interpolation='nearest', origin='upper')
        ax_terrain.set_title('Terrain Map (地形图)', fontsize=14, fontweight='bold', pad=15)
        ax_terrain.set_xlabel('X (East →)', fontsize=10)
        ax_terrain.set_ylabel('Y (South ↓)', fontsize=10)

        # 标注桥梁
        for bx, by in self.bridges:
            ax_terrain.plot(bx, by, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1)
            ax_terrain.text(bx, by - 1.5, 'Bridge', ha='center', fontsize=8,
                          color='red', fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        legend1 = [patches.Patch(facecolor=self.TERRAIN_COLORS[t], label=t.title())
                  for t in terrain_types]
        ax_terrain.legend(handles=legend1, loc='upper right', fontsize=9)

        # === 图2: 区域规划图 ===
        ax_zones = axes[1]
        zone_grid = np.zeros((self.height, self.width))
        zone_list = list(self.ZONE_COLORS.keys())
        zone_map = {z: i for i, z in enumerate(zone_list)}

        for y in range(self.height):
            for x in range(self.width):
                zone_grid[y, x] = zone_map.get(self.grid[y][x].zone_type, 0)

        colors = [self.ZONE_COLORS[z] for z in zone_list]
        cmap = ListedColormap(colors)

        ax_zones.imshow(zone_grid, cmap=cmap, interpolation='nearest', origin='upper')
        ax_zones.set_title('Zoning Plan (功能分区)', fontsize=14, fontweight='bold', pad=15)
        ax_zones.set_xlabel('X (East →)', fontsize=10)
        ax_zones.set_ylabel('Y (South ↓)', fontsize=10)

        legend2 = [patches.Patch(facecolor=self.ZONE_COLORS[z],
                                label=z.replace('_', ' ').title())
                  for z in ['residential', 'commercial', 'industrial', 'office', 'park', 'bridge']]
        ax_zones.legend(handles=legend2, loc='upper right', fontsize=8, ncol=2)

        # === 图3: 发展阶段图 ===
        ax_phases = axes[2]
        phase_grid = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x].zone_type != 'unplanned':
                    phase_grid[y, x] = self.grid[y][x].phase
                else:
                    phase_grid[y, x] = 0

        colors = [self.PHASE_COLORS[i] for i in [0, 1, 2, 3]]
        cmap = ListedColormap(colors)

        ax_phases.imshow(phase_grid, cmap=cmap, interpolation='nearest', origin='upper')
        ax_phases.set_title('Development Phases (发展阶段)', fontsize=14, fontweight='bold', pad=15)
        ax_phases.set_xlabel('X (East →)', fontsize=10)
        ax_phases.set_ylabel('Y (South ↓)', fontsize=10)

        # 显示阶段标注
        for y in range(0, self.height, 5):
            for x in range(0, self.width, 5):
                phase = self.grid[y][x].phase
                if phase > 0:
                    ax_phases.text(x, y, str(phase), ha='center', va='center',
                                 color='white', fontsize=10, fontweight='bold',
                                 bbox=dict(boxstyle='circle', facecolor='black', alpha=0.6))

        legend3 = [patches.Patch(facecolor=self.PHASE_COLORS[i],
                                label=f'Phase {i}' if i > 0 else 'Unplanned')
                  for i in [1, 2, 3, 0]]
        ax_phases.legend(handles=legend3, loc='upper right', fontsize=9)

        plt.tight_layout()
        return fig

    def print_plan(self):
        """打印详细规划"""
        print("\n" + "="*80)
        print("子母河地图初始规划 (Two Rivers Map - Initial Plan)")
        print("="*80)

        stats = self.get_statistics()

        print(f"\n网格尺寸: {self.width} × {self.height} = {stats['total_cells']} 格")
        print(f"可建设用地: {stats['buildable_count']} 格")
        print(f"河流水域: {stats['water_count']} 格")

        print("\n【阶段1 - 初期发展】(0-10k人口)")
        self._print_phase_zones(1)

        print("\n【阶段2 - 中期扩张】(10k-30k人口)")
        self._print_phase_zones(2)

        print("\n【阶段3 - 后期发展】(30k+人口)")
        self._print_phase_zones(3)

        print("\n【桥梁建设计划】")
        for i, (x, y) in enumerate(self.bridges, 1):
            print(f"  桥梁 {i}: 位置({x}, {y})")

    def _print_phase_zones(self, phase: int):
        """打印特定阶段的区域"""
        zones = {}
        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.phase == phase and cell.zone_type != 'unplanned':
                    key = cell.zone_type
                    if key not in zones:
                        zones[key] = []
                    zones[key].append((x, y, cell.notes))

        for zone_type, cells in sorted(zones.items()):
            print(f"  {zone_type.title()}: {len(cells)} 格")
            if cells[0][2]:  # 如果有备注
                print(f"    -> {cells[0][2]}")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'total_cells': self.width * self.height,
            'buildable_count': 0,
            'water_count': 0,
            'zones': {},
            'phases': {1: 0, 2: 0, 3: 0}
        }

        for y in range(self.height):
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.terrain == 'buildable':
                    stats['buildable_count'] += 1
                elif cell.terrain == 'water':
                    stats['water_count'] += 1

                if cell.zone_type != 'unplanned':
                    stats['zones'][cell.zone_type] = stats['zones'].get(cell.zone_type, 0) + 1
                    if cell.phase in stats['phases']:
                        stats['phases'][cell.phase] += 1

        return stats

    def save_json(self, filename: str):
        """保存到JSON"""
        data = {
            'map': 'Two Rivers (子母河)',
            'width': self.width,
            'height': self.height,
            'grid': [[cell.to_dict() for cell in row] for row in self.grid],
            'bridges': self.bridges,
            'statistics': self.get_statistics()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def create_two_rivers_initial_plan():
    """
    创建子母河地图的初始规划

    基于截图的地图特征：
    - 右侧有一条大河从上至下流淌
    - 左侧是大片可建设陆地
    - 右上角有高速公路出入口
    - 起始位置在左侧中部
    """

    planner = TwoRiversMapPlanner(width=40, height=40)

    # ========== 地形设置 ==========

    # 绘制主河道 (右侧大河)
    main_river = [(x, y) for x in range(28, 35) for y in range(40)]
    for x, y in main_river:
        planner.set_terrain(x, y, 'water')

    # 河流在中部略微弯曲
    for y in range(15, 25):
        planner.set_terrain(27, y, 'water')
        planner.set_terrain(26, y, 'water')

    # 高速公路 (右上角)
    for i in range(15):
        planner.set_terrain(35 + i // 3, i, 'highway')

    # 森林区域 (左上角)
    planner.set_area(0, 0, 8, 8, 'unplanned', phase=0)
    for y in range(9):
        for x in range(9):
            planner.set_terrain(x, y, 'forest')

    # ========== 阶段1: 初期发展 (优先级最高) ==========

    # 1. 起始住宅区 (左侧中部 - 接近高速出口)
    planner.set_area(5, 12, 12, 18, 'residential', priority=5, phase=1,
                     notes="起始住宅区 - 连接高速公路，优先发展")

    # 2. 初期商业区 (住宅区旁)
    planner.set_area(13, 12, 17, 16, 'commercial', priority=5, phase=1,
                     notes="商业街 - 服务起始住宅")

    # 3. 第一座桥梁 (连接河对岸)
    planner.add_bridge(26, 15)
    planner.set_zone(26, 15, 'bridge', priority=5, phase=1, notes="主桥 - 连接东西两岸")

    # 4. 轻工业区 (左下角 - 远离住宅)
    planner.set_area(2, 28, 10, 36, 'industrial', priority=4, phase=1,
                     notes="工业区 - 提供初期就业")

    # ========== 阶段2: 中期扩张 ==========

    # 5. 扩展住宅区 (向北发展)
    planner.set_area(5, 4, 14, 10, 'residential', priority=4, phase=2,
                     notes="北部住宅区 - 靠近森林，环境好")

    # 6. CBD办公区 (中心位置)
    planner.set_area(18, 12, 24, 18, 'office', priority=4, phase=2,
                     notes="CBD - 城市商务中心")

    # 7. 河对岸住宅区 (过桥后)
    planner.set_area(18, 19, 24, 26, 'residential', priority=3, phase=2,
                     notes="东岸住宅区 - 需要桥梁连接")

    # 8. 第二座桥梁 (南部连接)
    planner.add_bridge(26, 30)
    planner.set_zone(26, 30, 'bridge', priority=4, phase=2, notes="南桥 - 连接工业区")

    # 9. 滨河公园
    planner.set_area(18, 8, 24, 11, 'park', priority=3, phase=2,
                     notes="中央公园 - 提升城市品质")

    # ========== 阶段3: 后期发展 ==========

    # 10. 高档住宅区 (左下角剩余空间)
    planner.set_area(12, 28, 18, 36, 'residential', priority=2, phase=3,
                     notes="南部住宅区 - 后期扩展")

    # 11. 商业副中心 (南部)
    planner.set_area(19, 28, 24, 34, 'commercial', priority=2, phase=3,
                     notes="南部商业区")

    # 12. 混合用途区
    planner.set_area(15, 20, 18, 26, 'mixed', priority=2, phase=3,
                     notes="混合区 - 商住结合")

    return planner


if __name__ == "__main__":
    print("正在生成子母河地图初始规划...")

    # 创建规划
    planner = create_two_rivers_initial_plan()

    # 打印规划
    planner.print_plan()

    # 保存JSON
    planner.save_json('two_rivers_plan.json')
    print("\n✓ 规划数据已保存到 two_rivers_plan.json")

    # 生成可视化
    fig = planner.visualize()
    plt.savefig('two_rivers_plan.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 规划图已保存到 two_rivers_plan.png")

    # 不自动显示窗口，避免阻塞
    # plt.show()
    plt.close()

    print("\n" + "="*80)
    print("核心建议：")
    print("="*80)
    print("""
【起步策略】
1. 从左侧中部开始建设（靠近高速出口）
2. 先建住宅+商业，形成初始经济循环
3. 尽快建第一座桥连接河对岸
4. 工业区放在下风向（左下角）

【交通规划】
5. 主干道：左右横向连接各区
6. 次干道：南北纵向分流
7. 桥梁：至少2座（中部+南部）
8. 高速出口直连主干道

【资源管理】
9. 取水：从河流上游取水
10. 排污：下游排污，注意水流方向
11. 电力：初期用风电，后期考虑水电
12. 垃圾：工业区附近建垃圾场

【注意事项】
- 河流不可建设，必须用桥梁连接
- 左侧陆地面积大，是主要发展区
- 右侧河对岸可作为高档住宅区
- 预留公共交通走廊
    """)
